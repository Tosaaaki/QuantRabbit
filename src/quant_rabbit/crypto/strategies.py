from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .fast import BPS, FastMarketState, _d

DEFAULT_STRATEGY_CONFIG = Path("config/crypto_strategy_lab_v1.json")


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    family: str
    variant_of: str | None
    changed_category: str | None
    entry_order_style: str
    exit_order_style: str
    min_signal_bps: Decimal
    max_signal_bps: Decimal
    min_imbalance: Decimal
    max_spread_bps: Decimal
    gross_edge_multiplier: Decimal
    imbalance_edge_scale_bps: Decimal
    spread_cost_multiplier: Decimal
    adverse_selection_bps: Decimal
    safety_buffer_bps: Decimal
    take_profit_bps: Decimal
    stop_loss_bps: Decimal
    max_hold_ms: int
    cooldown_ms: int

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "StrategyProfile":
        order_styles = {
            str(raw.get("entry_order_style")),
            str(raw.get("exit_order_style")),
        }
        if not order_styles <= {"PAPER_MAKER_LIMIT", "PAPER_TAKER"}:
            raise ValueError(f"{name} has an invalid Paper order style")
        profile = cls(
            name=name,
            family=str(raw["family"]),
            variant_of=(
                str(raw["variant_of"]) if raw.get("variant_of") else None
            ),
            changed_category=(
                str(raw["changed_category"])
                if raw.get("changed_category")
                else None
            ),
            entry_order_style=str(raw["entry_order_style"]),
            exit_order_style=str(raw["exit_order_style"]),
            min_signal_bps=_d(raw["min_signal_bps"]),
            max_signal_bps=_d(raw["max_signal_bps"]),
            min_imbalance=_d(raw["min_imbalance"]),
            max_spread_bps=_d(raw["max_spread_bps"]),
            gross_edge_multiplier=_d(raw["gross_edge_multiplier"]),
            imbalance_edge_scale_bps=_d(
                raw["imbalance_edge_scale_bps"]
            ),
            spread_cost_multiplier=_d(raw["spread_cost_multiplier"]),
            adverse_selection_bps=_d(raw["adverse_selection_bps"]),
            safety_buffer_bps=_d(raw["safety_buffer_bps"]),
            take_profit_bps=_d(raw["take_profit_bps"]),
            stop_loss_bps=_d(raw["stop_loss_bps"]),
            max_hold_ms=int(raw["max_hold_ms"]),
            cooldown_ms=int(raw["cooldown_ms"]),
        )
        if (
            profile.min_signal_bps < 0
            or profile.max_signal_bps <= profile.min_signal_bps
            or not Decimal("0") <= profile.min_imbalance <= Decimal("1")
            or profile.max_spread_bps <= 0
            or profile.take_profit_bps <= 0
            or profile.stop_loss_bps <= 0
            or profile.max_hold_ms <= 0
            or profile.cooldown_ms < 0
        ):
            raise ValueError(f"{name} has invalid strategy bounds")
        return profile


def load_strategy_profiles(
    path: Path | None = None,
) -> dict[str, StrategyProfile]:
    config_path = path or Path(
        os.environ.get(
            "QR_CRYPTO_STRATEGY_CONFIG",
            str(DEFAULT_STRATEGY_CONFIG),
        )
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "QR_CRYPTO_STRATEGY_LAB_CONFIG_V1":
        raise ValueError("unsupported crypto strategy config")
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("crypto strategy config has no strategies")
    return {
        str(name): StrategyProfile.from_dict(str(name), dict(raw))
        for name, raw in strategies.items()
    }


class ConfiguredStrategyRouter:
    """Deterministic Paper-only sibling strategy using present/past data."""

    def __init__(
        self,
        profile: StrategyProfile,
        *,
        warmup_events: int = 12,
        book_levels: int = 8,
        max_data_age_ms: int = 3_000,
    ) -> None:
        self.profile = profile
        self.warmup_events = warmup_events
        self.book_levels = book_levels
        self.max_data_age_ms = max_data_age_ms
        self.last_intent_ns: dict[str, int] = {}
        self.opened_ns: dict[str, int] = {}

    def decide(
        self,
        state: FastMarketState,
        *,
        position: Decimal,
        average_cost: Decimal,
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
        allow_short: bool,
        now_ns: int,
        wall_time_ms: int,
    ) -> dict[str, Any]:
        features = state.book.features(self.book_levels)
        if not features:
            return self._wait(state.pair, "BOOK_NOT_READY")
        age_ms = wall_time_ms - state.book.published_at_ms
        if age_ms < -1_000:
            return self._wait(state.pair, "FUTURE_STREAM_DATA")
        if age_ms > self.max_data_age_ms:
            return self._wait(state.pair, "STALE_STREAM_DATA")
        if state.event_count < self.warmup_events:
            return self._wait(state.pair, "WARMUP")
        if len(state.prices) < 4:
            return self._wait(state.pair, "PRICE_HISTORY_SHORT")
        last_intent = self.last_intent_ns.get(state.pair, 0)
        if (
            now_ns - last_intent
            < self.profile.cooldown_ms * 1_000_000
        ):
            return self._wait(state.pair, "COOLDOWN")

        first = state.prices[0]
        last = state.prices[-1]
        prior = state.prices[-2]
        midpoint = state.prices[len(state.prices) // 2]
        momentum_bps = (
            (last - first) / first * BPS if first > 0 else Decimal("0")
        )
        recent_bps = (
            (last - prior) / prior * BPS if prior > 0 else Decimal("0")
        )
        second_half_bps = (
            (last - midpoint) / midpoint * BPS
            if midpoint > 0
            else Decimal("0")
        )
        spread_bps = features["spread_bps"]
        imbalance = features["imbalance"]
        signal = self._entry_signal(
            momentum_bps=momentum_bps,
            recent_bps=recent_bps,
            second_half_bps=second_half_bps,
            imbalance=imbalance,
        )
        gross_edge_bps = (
            signal["raw_edge_bps"] * self.profile.gross_edge_multiplier
            + signal["aligned_imbalance"]
            * self.profile.imbalance_edge_scale_bps
        )
        expected_cost_bps = self._expected_cost(
            spread_bps=spread_bps,
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
        )
        net_edge_bps = gross_edge_bps - expected_cost_bps
        common = {
            "pair": state.pair,
            "strategy": self.profile.name,
            "market_regime": signal["regime"],
            "momentum_bps": str(momentum_bps),
            "recent_bps": str(recent_bps),
            "spread_bps": str(spread_bps),
            "imbalance": str(imbalance),
            "expected_cost_bps": str(expected_cost_bps),
            "gross_edge_bps": str(gross_edge_bps),
            "net_edge_bps": str(net_edge_bps),
            "entry_order_style": self.profile.entry_order_style,
            "exit_order_style": self.profile.exit_order_style,
            "book_sequence": state.book.sequence,
            "authority": "NONE",
            "live_permission": False,
            "no_future_data": True,
        }
        if spread_bps > self.profile.max_spread_bps:
            return {**common, "action": "WAIT", "reason": "WIDE_SPREAD"}

        if position == 0:
            reason = str(signal["reason"])
            position_side = signal.get("position_side")
            if position_side is None:
                return {**common, "action": "WAIT", "reason": reason}
            if position_side == "SHORT" and not allow_short:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "SHORT_DISABLED",
                }
            if net_edge_bps <= self.profile.safety_buffer_bps:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "NET_EDGE_BELOW_BUFFER",
                }
            self.last_intent_ns[state.pair] = now_ns
            self.opened_ns[state.pair] = now_ns
            return {
                **common,
                "action": "ENTER",
                "position_side": position_side,
                "reason": reason,
            }

        pnl_bps = (
            (features["bid"] - average_cost) / average_cost * BPS
            if position > 0 and average_cost > 0
            else (average_cost - features["ask"]) / average_cost * BPS
            if position < 0 and average_cost > 0
            else Decimal("0")
        )
        opened_ns = self.opened_ns.setdefault(state.pair, now_ns)
        held_ms = (now_ns - opened_ns) // 1_000_000
        position_side = "LONG" if position > 0 else "SHORT"
        exit_reason: str | None = None
        if pnl_bps >= self.profile.take_profit_bps:
            exit_reason = "TAKE_PROFIT"
        elif pnl_bps <= -self.profile.stop_loss_bps:
            exit_reason = "STOP_LOSS"
        elif held_ms >= self.profile.max_hold_ms:
            exit_reason = "MAX_HOLD"
        elif self._invalidated(
            position_side,
            momentum_bps=momentum_bps,
            recent_bps=recent_bps,
            imbalance=imbalance,
        ):
            exit_reason = "SIGNAL_INVALIDATED"
        if exit_reason is not None:
            self.last_intent_ns[state.pair] = now_ns
            return {
                **common,
                "action": "EXIT",
                "position_side": position_side,
                "reason": exit_reason,
                "exit_order_style": (
                    self.profile.exit_order_style
                    if exit_reason == "TAKE_PROFIT"
                    else "PAPER_TAKER"
                ),
                "position_pnl_bps": str(pnl_bps),
                "held_ms": int(held_ms),
            }
        return {
            **common,
            "action": "WAIT",
            "position_side": position_side,
            "reason": "HOLD",
            "position_pnl_bps": str(pnl_bps),
            "held_ms": int(held_ms),
        }

    def _entry_signal(
        self,
        *,
        momentum_bps: Decimal,
        recent_bps: Decimal,
        second_half_bps: Decimal,
        imbalance: Decimal,
    ) -> dict[str, Any]:
        family = self.profile.family
        magnitude = abs(momentum_bps)
        if family == "RANGE_MAKER_REVERSION":
            if not self.profile.min_signal_bps <= magnitude <= (
                self.profile.max_signal_bps
            ):
                return self._no_signal("RANGE_MOVE_OUTSIDE_BAND", "RANGE")
            side = "SHORT" if momentum_bps > 0 else "LONG"
            aligned = -imbalance if side == "SHORT" else imbalance
            if aligned < self.profile.min_imbalance:
                return self._no_signal(
                    "REVERSION_NOT_CONFIRMED", "RANGE"
                )
            return self._signal(
                side, magnitude, aligned, "RANGE_REVERSION", "RANGE"
            )
        if family == "BREAKOUT_CONFIRMATION":
            if magnitude < self.profile.min_signal_bps:
                return self._no_signal("BREAKOUT_TOO_SMALL", "RANGE")
            side = "LONG" if momentum_bps > 0 else "SHORT"
            aligned = imbalance if side == "LONG" else -imbalance
            if (
                aligned < self.profile.min_imbalance
                or recent_bps * momentum_bps <= 0
            ):
                return self._no_signal(
                    "BREAKOUT_NOT_CONFIRMED", "BREAKOUT_PENDING"
                )
            return self._signal(
                side,
                magnitude,
                aligned,
                "BREAKOUT_CONFIRMED",
                "TREND_UP" if side == "LONG" else "TREND_DOWN",
            )
        if family == "TREND_PULLBACK_MAKER":
            if magnitude < self.profile.min_signal_bps:
                return self._no_signal("TREND_TOO_SMALL", "RANGE")
            side = "LONG" if momentum_bps > 0 else "SHORT"
            is_pullback = (
                recent_bps < 0 if side == "LONG" else recent_bps > 0
            )
            resumed = (
                second_half_bps > 0
                if side == "LONG"
                else second_half_bps < 0
            )
            aligned = imbalance if side == "LONG" else -imbalance
            if not is_pullback or not resumed:
                return self._no_signal(
                    "PULLBACK_NOT_READY",
                    "TREND_UP" if side == "LONG" else "TREND_DOWN",
                )
            if aligned < -self.profile.min_imbalance:
                return self._no_signal(
                    "PULLBACK_BOOK_ADVERSE",
                    "TREND_UP" if side == "LONG" else "TREND_DOWN",
                )
            return self._signal(
                side,
                magnitude,
                max(Decimal("0"), aligned),
                "TREND_PULLBACK",
                "TREND_UP" if side == "LONG" else "TREND_DOWN",
            )
        if family == "ORDER_BOOK_FADE":
            if (
                magnitude > self.profile.max_signal_bps
                or abs(imbalance) < self.profile.min_imbalance
            ):
                return self._no_signal(
                    "FADE_CONDITIONS_NOT_MET", "RANGE"
                )
            side = "SHORT" if imbalance > 0 else "LONG"
            return self._signal(
                side,
                max(magnitude, self.profile.min_signal_bps),
                abs(imbalance),
                "EXTREME_BOOK_FADE",
                "RANGE",
            )
        raise ValueError(f"unsupported strategy family: {family}")

    def _invalidated(
        self,
        position_side: str,
        *,
        momentum_bps: Decimal,
        recent_bps: Decimal,
        imbalance: Decimal,
    ) -> bool:
        family = self.profile.family
        if family in {"RANGE_MAKER_REVERSION", "ORDER_BOOK_FADE"}:
            return (
                momentum_bps > self.profile.max_signal_bps
                if position_side == "SHORT"
                else momentum_bps < -self.profile.max_signal_bps
            )
        if family == "BREAKOUT_CONFIRMATION":
            return (
                recent_bps < 0 or imbalance < 0
                if position_side == "LONG"
                else recent_bps > 0 or imbalance > 0
            )
        return recent_bps * (Decimal("1") if position_side == "LONG" else -1) > 0

    def _expected_cost(
        self,
        *,
        spread_bps: Decimal,
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
    ) -> Decimal:
        def fee(style: str) -> Decimal:
            rate = (
                maker_fee_rate
                if style == "PAPER_MAKER_LIMIT"
                else taker_fee_rate
            )
            return max(Decimal("0"), rate * BPS)

        return (
            fee(self.profile.entry_order_style)
            + fee(self.profile.exit_order_style)
            + spread_bps * self.profile.spread_cost_multiplier
            + self.profile.adverse_selection_bps
        )

    def _wait(self, pair: str, reason: str) -> dict[str, Any]:
        return {
            "pair": pair,
            "strategy": self.profile.name,
            "action": "WAIT",
            "reason": reason,
            "authority": "NONE",
            "live_permission": False,
            "no_future_data": True,
        }

    @staticmethod
    def _no_signal(reason: str, regime: str) -> dict[str, Any]:
        return {
            "position_side": None,
            "raw_edge_bps": Decimal("0"),
            "aligned_imbalance": Decimal("0"),
            "reason": reason,
            "regime": regime,
        }

    @staticmethod
    def _signal(
        side: str,
        raw_edge_bps: Decimal,
        aligned_imbalance: Decimal,
        reason: str,
        regime: str,
    ) -> dict[str, Any]:
        return {
            "position_side": side,
            "raw_edge_bps": raw_edge_bps,
            "aligned_imbalance": max(
                Decimal("0"), aligned_imbalance
            ),
            "reason": reason,
            "regime": regime,
        }


def strategy_router(
    name: str,
    *,
    config_path: Path | None = None,
    warmup_events: int = 12,
    book_levels: int = 8,
    max_data_age_ms: int = 3_000,
) -> ConfiguredStrategyRouter:
    profiles = load_strategy_profiles(config_path)
    try:
        profile = profiles[name]
    except KeyError as exc:
        raise ValueError(f"unknown crypto Paper strategy: {name}") from exc
    return ConfiguredStrategyRouter(
        profile,
        warmup_events=warmup_events,
        book_levels=book_levels,
        max_data_age_ms=max_data_age_ms,
    )
