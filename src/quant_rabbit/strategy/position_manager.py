from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Side
from quant_rabbit.paths import (
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_TRADER_DECISION,
)
from quant_rabbit.risk import RiskPolicy
from quant_rabbit.strategy.intent_generator import (
    GEOMETRY_ATR_TIMEFRAME,
    GEOMETRY_SPREAD_FLOOR_MULT,
    _atr_pips_for,
    _load_pair_charts,
    _session_bucket_for,
)


ACTION_HOLD_PROTECTED = "HOLD_PROTECTED"
ACTION_HOLD_SL_FREE = "HOLD_SL_FREE"
ACTION_PROFIT_PROTECT = "PROFIT_PROTECT_REQUIRED"
ACTION_REVIEW_EXIT = "REVIEW_EXIT"
ACTION_REPAIR_PROTECTION = "REPAIR_PROTECTION_REQUIRED"


# When `QR_TRADER_DISABLE_SL_REPAIR=1` the protection gateway treats a missing
# SL on a trader-owned position as deliberate (the user's "SLいらない" directive,
# `feedback_no_tight_sl_thin_market.md`) and emits HOLD_SL_FREE instead of
# REPAIR_PROTECTION_REQUIRED. TP repair, profit protection, and contradiction
# exits still apply.
def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}

# Profit protection must not move SL to breakeven while the market is still
# inside ordinary execution noise. Use the same spread floor as entry geometry
# plus one current M5 ATR: this is market-derived noise room, not a profit gate.
PROFIT_PROTECTION_NOISE_ATR_MULT = 1.0
PROFIT_PROTECTION_SPREAD_MULT = GEOMETRY_SPREAD_FLOOR_MULT


@dataclass(frozen=True)
class ManagedPosition:
    trade_id: str
    pair: str
    side: str
    units: int
    action: str
    unrealized_pl_jpy: float
    remaining_risk_jpy: float | None
    remaining_reward_jpy: float | None
    same_direction_score: float | None
    opposite_direction_score: float | None
    recommended_stop_loss: float | None
    recommended_take_profit: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PositionManagementDecision:
    generated_at_utc: str
    action: str
    positions: tuple[ManagedPosition, ...]


class PositionManager:
    """Manage open exposure as a trader decision, not a passive monitor."""

    def __init__(
        self,
        *,
        trader_decision_path: Path = DEFAULT_TRADER_DECISION,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        output_path: Path = DEFAULT_POSITION_MANAGEMENT,
        report_path: Path = DEFAULT_POSITION_MANAGEMENT_REPORT,
    ) -> None:
        self.trader_decision_path = trader_decision_path
        self.pair_charts_path = pair_charts_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self, snapshot: BrokerSnapshot) -> PositionManagementDecision:
        generated_at = datetime.now(timezone.utc).isoformat()
        scores = _load_scores(self.trader_decision_path)
        pair_charts = _load_pair_charts(self.pair_charts_path)
        trader_positions = tuple(position for position in snapshot.positions if position.owner == Owner.TRADER)
        managed = tuple(self._manage_position(position, snapshot, scores, pair_charts) for position in trader_positions)
        action = _aggregate_action(managed)
        decision = PositionManagementDecision(generated_at_utc=generated_at, action=action, positions=managed)
        self._write(decision)
        return decision

    def _manage_position(
        self,
        position: BrokerPosition,
        snapshot: BrokerSnapshot,
        scores: dict[tuple[str, str], float],
        pair_charts: dict[str, dict[str, Any]] | None,
    ) -> ManagedPosition:
        same_score = scores.get((position.pair, position.side.value))
        opposite_score = scores.get((position.pair, _opposite(position.side)))
        remaining_risk = _remaining_risk_jpy(position, snapshot.quotes, snapshot.home_conversions)
        remaining_reward = _remaining_reward_jpy(position, snapshot.quotes, snapshot.home_conversions)
        reasons: list[str] = []
        quote = snapshot.quotes.get(position.pair)
        recommended_stop_loss: float | None = None
        recommended_take_profit: float | None = None
        reasons.extend(_session_protection_notes(position, quote, pair_charts))

        sl_free_hold = (
            position.stop_loss is None
            and position.owner == Owner.TRADER
            and _trader_sl_repair_disabled()
        )
        if position.stop_loss is None or position.take_profit is None:
            missing = []
            if position.take_profit is None:
                missing.append("TP")
            if position.stop_loss is None:
                missing.append("SL")
            reasons.append(f"missing {'/'.join(missing)}")
            if position.stop_loss is None:
                if sl_free_hold:
                    reasons.append("trader SL-repair disabled (QR_TRADER_DISABLE_SL_REPAIR=1); discretionary SL-free hold")
                    action = ACTION_HOLD_SL_FREE
                else:
                    recommended_stop_loss = _repair_stop_loss(position, quote, snapshot.quotes, snapshot.home_conversions)
                    if recommended_stop_loss is None:
                        reasons.append("no market-valid capped SL repair is available; exposure needs exit review")
                        action = ACTION_REVIEW_EXIT
                    else:
                        reasons.append(f"repair SL candidate {recommended_stop_loss:.5f}")
                        action = ACTION_REPAIR_PROTECTION
            else:
                action = ACTION_REPAIR_PROTECTION
            if position.take_profit is None:
                basis_stop = recommended_stop_loss if recommended_stop_loss is not None else position.stop_loss
                recommended_take_profit = _repair_take_profit(position, basis_stop, quote)
                if recommended_take_profit is not None:
                    reasons.append(f"repair TP candidate {recommended_take_profit:.5f}")
        else:
            profit_protection_needed, profit_reasons = _profit_protection_needed(
                position,
                remaining_risk,
                quote,
                snapshot.quotes,
                snapshot.home_conversions,
                pair_charts,
            )
            reasons.extend(profit_reasons)
            if profit_protection_needed:
                reasons.append("profit clears remaining risk plus current session noise")
                recommended_stop_loss = _break_even_stop(position, quote)
                if recommended_stop_loss is None:
                    reasons.append("break-even SL is not market-valid yet")
                else:
                    reasons.append(f"break-even SL candidate {recommended_stop_loss:.5f}")
                action = ACTION_PROFIT_PROTECT
            elif opposite_score is not None and same_score is not None and opposite_score >= same_score + 20 and position.unrealized_pl_jpy < 0:
                reasons.append(f"opposite thesis score {opposite_score:.1f} materially exceeds same-direction {same_score:.1f}")
                action = ACTION_REVIEW_EXIT
            else:
                reasons.append("TP/SL present and current thesis is not contradicted enough to force exit")
                action = ACTION_HOLD_PROTECTED

        if remaining_risk is not None:
            reasons.append(f"remaining risk about {remaining_risk:.0f} JPY")
        elif position.stop_loss is not None:
            reasons.append("remaining risk cannot be converted to JPY from current broker snapshot")
        if remaining_reward is not None:
            reasons.append(f"remaining reward about {remaining_reward:.0f} JPY")
        elif position.take_profit is not None:
            reasons.append("remaining reward cannot be converted to JPY from current broker snapshot")

        return ManagedPosition(
            trade_id=position.trade_id,
            pair=position.pair,
            side=position.side.value,
            units=position.units,
            action=action,
            unrealized_pl_jpy=round(position.unrealized_pl_jpy, 4),
            remaining_risk_jpy=round(remaining_risk, 2) if remaining_risk is not None else None,
            remaining_reward_jpy=round(remaining_reward, 2) if remaining_reward is not None else None,
            same_direction_score=same_score,
            opposite_direction_score=opposite_score,
            recommended_stop_loss=round(recommended_stop_loss, _price_precision(position.pair))
            if recommended_stop_loss is not None
            else None,
            recommended_take_profit=round(recommended_take_profit, _price_precision(position.pair))
            if recommended_take_profit is not None
            else None,
            reasons=tuple(reasons),
        )

    def _write(self, decision: PositionManagementDecision) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Position Management Report",
            "",
            f"- Generated at UTC: `{decision.generated_at_utc}`",
            f"- Action: `{decision.action}`",
            f"- Positions: `{len(decision.positions)}`",
            "",
            "## Positions",
            "",
        ]
        if not decision.positions:
            lines.append("- none")
        for item in decision.positions:
            lines.append(
                f"- `{item.trade_id}` `{item.pair} {item.side}` units=`{item.units}` "
                f"action=`{item.action}` upl=`{item.unrealized_pl_jpy:.1f}`"
            )
            lines.append(f"  - scores: same=`{item.same_direction_score}` opposite=`{item.opposite_direction_score}`")
            lines.append(
                f"  - protection plan: sl=`{item.recommended_stop_loss}` tp=`{item.recommended_take_profit}`"
            )
            for reason in item.reasons:
                lines.append(f"  - reason: {reason}")
        lines.extend(
            [
                "",
                "## Management Contract",
                "",
                "- Existing positions are managed before any new entry is considered.",
                "- Operator-managed manual/tagless positions are observed in broker truth but ignored by this gateway.",
                "- Missing TP/SL is a repair requirement, not a passive monitor state.",
                "- Profit protection is required once open profit clears remaining stop risk plus current session noise.",
                "- A materially stronger opposite thesis triggers exit review; the gateway still prevents fresh stacking.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _load_scores(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    scores: dict[tuple[str, str], float] = {}
    for item in payload.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        if not pair or not direction:
            continue
        key = (pair, direction)
        score = float(item.get("score") or 0.0)
        if score > scores.get(key, -10_000.0):
            scores[key] = score
    return scores


def _aggregate_action(positions: tuple[ManagedPosition, ...]) -> str:
    actions = {position.action for position in positions}
    if ACTION_REPAIR_PROTECTION in actions:
        return ACTION_REPAIR_PROTECTION
    if ACTION_REVIEW_EXIT in actions:
        return ACTION_REVIEW_EXIT
    if ACTION_PROFIT_PROTECT in actions:
        return ACTION_PROFIT_PROTECT
    if positions:
        return ACTION_HOLD_PROTECTED
    return "NO_POSITION"


def _remaining_risk_jpy(position: BrokerPosition, quotes, home_conversions=None) -> float | None:
    if position.stop_loss is None:
        return None
    pips = (position.entry_price - position.stop_loss) * _pip_factor(position.pair)
    if position.side == Side.SHORT:
        pips = (position.stop_loss - position.entry_price) * _pip_factor(position.pair)
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None:
        return None
    return max(0.0, pips) * jpy_per_pip


def _remaining_reward_jpy(position: BrokerPosition, quotes, home_conversions=None) -> float | None:
    if position.take_profit is None:
        return None
    pips = (position.take_profit - position.entry_price) * _pip_factor(position.pair)
    if position.side == Side.SHORT:
        pips = (position.entry_price - position.take_profit) * _pip_factor(position.pair)
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None:
        return None
    return max(0.0, pips) * jpy_per_pip


def _profit_protection_needed(
    position: BrokerPosition,
    remaining_risk: float | None,
    quote,
    quotes,
    home_conversions,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[bool, tuple[str, ...]]:
    if position.stop_loss is None:
        return True, ("profit protection requires a stop loss first",)
    if _stop_is_break_even_or_better(position):
        return False, ("SL already at/through break-even",)
    if remaining_risk is None or remaining_risk <= 0:
        return False, ("profit protection deferred: remaining risk cannot be measured",)

    noise_jpy = _profit_protection_noise_jpy(position, quote, quotes, home_conversions, pair_charts)
    if noise_jpy is None:
        return False, ("profit protection deferred until session ATR/spread noise can be measured",)

    threshold = remaining_risk + noise_jpy
    if position.unrealized_pl_jpy < threshold:
        return (
            False,
            (
                f"profit protection deferred: upl {position.unrealized_pl_jpy:.0f} JPY < "
                f"remaining risk {remaining_risk:.0f} + session noise {noise_jpy:.0f}",
            ),
        )
    return (
        True,
        (
            f"profit protection trigger: upl {position.unrealized_pl_jpy:.0f} JPY >= "
            f"remaining risk {remaining_risk:.0f} + session noise {noise_jpy:.0f}",
        ),
    )


def _profit_protection_noise_jpy(
    position: BrokerPosition,
    quote,
    quotes,
    home_conversions,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> float | None:
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None or jpy_per_pip <= 0:
        return None
    noise_pips: list[float] = []
    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is not None and atr_pips > 0:
        noise_pips.append(atr_pips * PROFIT_PROTECTION_NOISE_ATR_MULT)
    spread_pips = _spread_pips(position.pair, quote)
    if spread_pips is not None and spread_pips > 0:
        noise_pips.append(spread_pips * PROFIT_PROTECTION_SPREAD_MULT)
    if not noise_pips:
        return None
    return max(noise_pips) * jpy_per_pip


def _session_protection_notes(
    position: BrokerPosition,
    quote,
    pair_charts: dict[str, dict[str, Any]] | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    session_bucket = _session_bucket_for(position.pair, pair_charts)
    if session_bucket:
        notes.append(f"session bucket {session_bucket}")
    else:
        notes.append("session bucket unavailable")
    atr_pips = _atr_pips_for(position.pair, pair_charts, GEOMETRY_ATR_TIMEFRAME)
    if atr_pips is not None:
        notes.append(f"{GEOMETRY_ATR_TIMEFRAME} ATR about {atr_pips:.1f}pip")
    else:
        notes.append(f"{GEOMETRY_ATR_TIMEFRAME} ATR unavailable for session SL/TP noise evaluation")
    spread_pips = _spread_pips(position.pair, quote)
    if spread_pips is not None:
        notes.append(f"current spread about {spread_pips:.1f}pip")
    sl_pips = _position_sl_pips(position)
    if sl_pips is not None:
        if atr_pips is not None and atr_pips > 0:
            notes.append(f"SL distance {sl_pips:.1f}pip = {sl_pips / atr_pips:.1f}x {GEOMETRY_ATR_TIMEFRAME} ATR")
        else:
            notes.append(f"SL distance {sl_pips:.1f}pip")
    tp_pips = _position_tp_pips(position)
    if tp_pips is not None:
        if atr_pips is not None and atr_pips > 0:
            notes.append(f"TP distance {tp_pips:.1f}pip = {tp_pips / atr_pips:.1f}x {GEOMETRY_ATR_TIMEFRAME} ATR")
        else:
            notes.append(f"TP distance {tp_pips:.1f}pip")
    return tuple(notes)


def _stop_is_break_even_or_better(position: BrokerPosition) -> bool:
    if position.stop_loss is None:
        return False
    if position.side == Side.LONG:
        return position.stop_loss >= position.entry_price
    return position.stop_loss <= position.entry_price


def _position_sl_pips(position: BrokerPosition) -> float | None:
    if position.stop_loss is None:
        return None
    distance = position.entry_price - position.stop_loss if position.side == Side.LONG else position.stop_loss - position.entry_price
    return max(0.0, distance * _pip_factor(position.pair))


def _position_tp_pips(position: BrokerPosition) -> float | None:
    if position.take_profit is None:
        return None
    distance = position.take_profit - position.entry_price if position.side == Side.LONG else position.entry_price - position.take_profit
    return max(0.0, distance * _pip_factor(position.pair))


def _spread_pips(pair: str, quote) -> float | None:
    if quote is None:
        return None
    spread = (quote.ask - quote.bid) * _pip_factor(pair)
    return spread if spread > 0 else None


def _break_even_stop(position: BrokerPosition, quote) -> float | None:
    if quote is None:
        return position.entry_price
    if position.side == Side.LONG:
        if quote.bid <= position.entry_price:
            return None
        return position.entry_price
    if quote.ask >= position.entry_price:
        return None
    return position.entry_price


def _repair_stop_loss(position: BrokerPosition, quote, quotes, home_conversions=None) -> float | None:
    jpy_per_pip = _jpy_per_pip(position, quotes, home_conversions or {})
    if jpy_per_pip is None or jpy_per_pip <= 0:
        return None
    cap = _repair_loss_cap_jpy()
    if cap is None:
        return None
    cap_pips = cap / jpy_per_pip
    repair_pips = min(cap_pips, _default_repair_stop_pips(position.pair))
    distance = repair_pips / _pip_factor(position.pair)
    candidate = position.entry_price - distance if position.side == Side.LONG else position.entry_price + distance
    if quote is None:
        return candidate
    if not _market_valid_stop(position, candidate, quote):
        return None
    return candidate


def _repair_take_profit(position: BrokerPosition, stop_loss: float | None, quote) -> float | None:
    if stop_loss is None:
        return None
    risk_distance = abs(position.entry_price - stop_loss)
    if risk_distance <= 0:
        return None
    candidate = (
        position.entry_price + risk_distance * 1.5
        if position.side == Side.LONG
        else position.entry_price - risk_distance * 1.5
    )
    if quote is None:
        return candidate
    if position.side == Side.LONG and candidate <= quote.ask:
        return None
    if position.side == Side.SHORT and candidate >= quote.bid:
        return None
    return candidate


def _market_valid_stop(position: BrokerPosition, stop_loss: float, quote) -> bool:
    if position.side == Side.LONG:
        return stop_loss < quote.bid
    return stop_loss > quote.ask


def _default_repair_stop_pips(pair: str) -> float:
    return 10.0 if pair.endswith("_JPY") else 8.0


def _opposite(side: Side) -> str:
    return Side.SHORT.value if side == Side.LONG else Side.LONG.value


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _price_precision(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _jpy_per_pip(position: BrokerPosition, quotes, home_conversions) -> float | None:
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


def _repair_loss_cap_jpy() -> float | None:
    """Return the current per-trade cap for capped SL repair.

    Position repair must not widen exposure from a stale literal. Prefer the
    daily target ledger's equity-derived per-trade cap; use RiskPolicy's
    documented library default only when the ledger is absent in tests/ad-hoc
    runs.
    """
    if DEFAULT_DAILY_TARGET_STATE.exists():
        try:
            payload = json.loads(DEFAULT_DAILY_TARGET_STATE.read_text())
            value = float(payload.get("per_trade_risk_budget_jpy") or 0.0)
            if value > 0:
                return value
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    policy_cap = RiskPolicy().max_loss_jpy
    return float(policy_cap) if policy_cap is not None and policy_cap > 0 else None
