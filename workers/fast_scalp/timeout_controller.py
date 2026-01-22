"""
Advanced timeout and exit controller for the FastScalp worker.

The controller keeps per-trade state and evaluates:
    - Event-budget driven exits
    - Grace-period scratch checks
    - Hazard comparison (TP vs SL lead probabilities)
    - Adaptive wall-clock timeout windows

It surfaces exit reasons back to the worker loop while retaining summary
statistics that are logged once a trade is closed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from . import config
from .signal import SignalFeatures
from utils.tuning_loader import get_tuning_value


@dataclass(slots=True)
class TimeoutDecision:
    action: str = "none"  # "none" | "close"
    reason: Optional[str] = None


@dataclass(slots=True)
class TradeTimeoutState:
    trade_id: str
    side: str  # "long" | "short"
    entry_price: float
    entry_monotonic: float
    signal_strength: float
    entry_slip_pips: float
    event_budget_base: int
    event_budget_current: int
    timeout_sec_base: float
    timeout_sec_current: float
    extra_timeout_sec: float = 0.0
    events_used: int = 0
    grace_used: bool = False
    scratch_attempted: bool = False
    scratch_events: int = 0
    hazard_counter: int = 0
    hazard_triggered: bool = False
    last_health: float = 0.0
    health_min: float = 1.0
    p_tp_min: float = 1.0
    p_tp_last: float = 0.5
    cost_k_last: float = 1.0
    last_tick_rate: float = 0.0
    last_latency_ms: float = 0.0
    last_update_monotonic: float = field(default_factory=time.monotonic)
    last_extend_event: int = -1


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class TimeoutController:
    def __init__(self) -> None:
        self._trades: Dict[str, TradeTimeoutState] = {}

    # --- registration helpers -------------------------------------------------
    def register_trade(
        self,
        trade_id: str,
        *,
        side: str,
        entry_price: float,
        entry_monotonic: float,
        features: Optional[SignalFeatures],
        spread_pips: float,
        tick_rate: float,
        latency_ms: Optional[float] = None,
    ) -> None:
        if trade_id in self._trades:
            return
        signal_strength = self._estimate_signal_strength(side, features)
        entry_slip = self._estimate_entry_slip(entry_price, features)
        tick_rate_eff = tick_rate if tick_rate > 0 else self._estimate_tick_rate(features)
        spread_eff = spread_pips if spread_pips > 0 else (features.spread_pips if features else 0.3)
        latency_ms_eff = latency_ms if latency_ms is not None else 160.0
        vola_ratio = self._resolve_vola_ratio(features)
        timeout_sec = self._adaptive_timeout_sec(
            tick_rate=tick_rate_eff,
            spread=spread_eff,
            latency_ms=latency_ms_eff,
            vola_ratio=vola_ratio,
            ai_strength=signal_strength,
            extra_sec=0.0,
        )
        state = TradeTimeoutState(
            trade_id=trade_id,
            side=side,
            entry_price=entry_price,
            entry_monotonic=entry_monotonic,
            signal_strength=signal_strength,
            entry_slip_pips=entry_slip,
            event_budget_base=config.TIMEOUT_EVENT_BUDGET,
            event_budget_current=config.TIMEOUT_EVENT_BUDGET,
            timeout_sec_base=timeout_sec,
            timeout_sec_current=timeout_sec,
        )
        state.last_tick_rate = tick_rate_eff
        state.last_latency_ms = latency_ms_eff
        self._trades[trade_id] = state

    def has_trade(self, trade_id: str) -> bool:
        return trade_id in self._trades

    # --- public API -----------------------------------------------------------
    def update(
        self,
        trade_id: str,
        *,
        elapsed_sec: float,
        pips_gain: float,
        features: SignalFeatures,
        tick_rate: float,
        latency_ms: Optional[float],
    ) -> TimeoutDecision:
        state = self._trades.get(trade_id)
        if state is None:
            return TimeoutDecision()

        monotonic_now = time.monotonic()
        latency_ms_val = latency_ms if latency_ms is not None else state.last_latency_ms
        state.last_latency_ms = latency_ms_val
        state.last_update_monotonic = monotonic_now

        elapsed_ms = elapsed_sec * 1000.0
        grace_ms = float(
            get_tuning_value(
                ("exit", "lowvol", "min_grace_before_scratch_ms"),
                config.TIMEOUT_GRACE_MS,
            )
            or config.TIMEOUT_GRACE_MS
        )
        in_grace = elapsed_ms < grace_ms

        if in_grace:
            state.grace_used = True
            state.scratch_events += 1
            scratch_min_events = int(
                float(
                    get_tuning_value(
                        ("exit", "lowvol", "scratch_requires_events"),
                        config.SCRATCH_REQUIRES_EVENTS,
                    )
                    or config.SCRATCH_REQUIRES_EVENTS
                )
            )
            if (
                not state.scratch_attempted
                and state.scratch_events >= scratch_min_events
                and self._should_scratch(state, features, latency_ms_val)
            ):
                state.scratch_attempted = True
                return TimeoutDecision(action="close", reason="scratch_early_no_support")
            # Grace期間中はイベントカウントを進めない
            return TimeoutDecision()

        state.events_used += 1
        state.last_tick_rate = tick_rate

        health = self._compute_health(state, features, pips_gain, tick_rate, elapsed_sec)
        state.last_health = health
        state.health_min = min(state.health_min, health)

        # 即時撤退判定
        if health <= config.TIMEOUT_HEALTH_KILL_THRESHOLD:
            return TimeoutDecision(action="close", reason="health_exit")

        # ハザード比較
        p_tp, p_sl, cost_k, hazard_hit = self._hazard_check(state, features, tick_rate, latency_ms_val)
        state.p_tp_last = p_tp
        state.cost_k_last = cost_k
        state.p_tp_min = min(state.p_tp_min, p_tp)
        if hazard_hit:
            state.hazard_counter += 1
            hazard_debounce = int(
                float(
                    get_tuning_value(
                        ("exit", "lowvol", "hazard_debounce_ticks"),
                        config.HAZARD_DEBOUNCE_EVENTS,
                    )
                    or config.HAZARD_DEBOUNCE_EVENTS
                )
            )
            if state.hazard_counter >= max(1, hazard_debounce):
                state.hazard_triggered = True
                return TimeoutDecision(action="close", reason="hazard_exit")
        else:
            state.hazard_counter = 0

        # 健康度に応じたイベント延命
        if (
            health >= config.TIMEOUT_HEALTH_EXTEND_THRESHOLD
            and state.event_budget_current < config.TIMEOUT_EVENT_BUDGET_MAX
            and state.last_extend_event != state.events_used
        ):
            state.event_budget_current = min(
                config.TIMEOUT_EVENT_BUDGET_MAX,
                state.event_budget_current + config.TIMEOUT_EVENT_EXTEND_EVENTS,
            )
            state.extra_timeout_sec = min(
                state.extra_timeout_sec + config.TIMEOUT_EVENT_EXTEND_SEC,
                max(0.0, config.TIMEOUT_ADAPTIVE_MAX_SEC - state.timeout_sec_base),
            )
            state.last_extend_event = state.events_used

        # イベント予算の枯渇
        if (
            state.events_used >= state.event_budget_current
            and health <= config.TIMEOUT_EVENT_HEALTH_EXIT
        ):
            return TimeoutDecision(action="close", reason="event_budget_timeout")

        # ハードタイムアウト
        vola_ratio = self._resolve_vola_ratio(features)
        adaptive_timeout = self._adaptive_timeout_sec(
            tick_rate=tick_rate,
            spread=features.spread_pips,
            latency_ms=latency_ms_val,
            vola_ratio=vola_ratio,
            ai_strength=state.signal_strength,
            extra_sec=state.extra_timeout_sec,
        )
        state.timeout_sec_current = adaptive_timeout
        if elapsed_sec >= adaptive_timeout:
            return TimeoutDecision(action="close", reason="hard_timeout")

        return TimeoutDecision()

    def finalize(
        self,
        trade_id: str,
        *,
        reason: str,
        pips_gain: float,
        tick_rate: float,
        spread_pips: float,
    ) -> Dict[str, float | str | bool]:
        state = self._trades.pop(trade_id, None)
        if state is None:
            return {}
        timeout_type = ""
        if reason in {"hard_timeout", "event_budget_timeout"}:
            timeout_type = self._classify_timeout(
                pips_gain=pips_gain,
                tick_rate=tick_rate,
                entry_slip=state.entry_slip_pips,
                spread=spread_pips,
            )
        summary: Dict[str, float | str | bool] = {
            "reason": reason,
            "events_used": state.events_used,
            "event_budget": state.event_budget_current,
            "timeout_sec": round(state.timeout_sec_current, 3),
            "health_min": round(state.health_min, 3),
            "health_last": round(state.last_health, 3),
            "p_tp_min": round(state.p_tp_min, 3),
            "p_tp_last": round(state.p_tp_last, 3),
            "hazard_debounce": state.hazard_counter,
            "hazard_triggered": state.hazard_triggered,
            "grace_used": state.grace_used,
            "entry_slip": round(state.entry_slip_pips, 3),
            "tick_rate_last": round(state.last_tick_rate, 3),
            "latency_ms_last": round(state.last_latency_ms, 1),
            "cost_k_last": round(state.cost_k_last, 3),
            "timeout_type": timeout_type,
            "pnl_pips": round(pips_gain, 3),
        }
        return summary

    # --- internal helpers -----------------------------------------------------
    def _estimate_signal_strength(self, side: str, features: Optional[SignalFeatures]) -> float:
        if not features:
            return 0.7
        directional_mom = features.short_momentum_pips if side == "long" else -features.short_momentum_pips
        strength = _clamp(abs(directional_mom) / 0.45, 0.0, 1.0)
        return max(0.55, strength)

    def _estimate_entry_slip(self, entry_price: float, features: Optional[SignalFeatures]) -> float:
        if not features:
            return 0.0
        return abs(features.latest_mid - entry_price) / config.PIP_VALUE

    def _estimate_tick_rate(self, features: Optional[SignalFeatures]) -> float:
        if not features or features.span_seconds <= 0.1:
            return 8.0
        return features.tick_count / max(0.5, features.span_seconds)

    def _resolve_vola_ratio(self, features: Optional[SignalFeatures]) -> float:
        if not features or features.atr_pips is None:
            return 1.0
        base = max(config.ATR_LOW_VOL_PIPS, 0.1)
        return _clamp(features.atr_pips / base, 0.6, 1.8)

    def _should_scratch(
        self,
        state: TradeTimeoutState,
        features: SignalFeatures,
        latency_ms: float,
    ) -> bool:
        directional_mom = features.short_momentum_pips if state.side == "long" else -features.short_momentum_pips
        imbalance = features.momentum_pips if state.side == "long" else -features.momentum_pips
        if features.spread_pips > config.SCRATCH_MAX_SPREAD:
            return True
        if directional_mom < config.SCRATCH_MOMENTUM_MIN:
            return True
        if imbalance < config.SCRATCH_IMBALANCE_MIN:
            return True
        if latency_ms > 450.0:
            return True
        return False

    def _compute_health(
        self,
        state: TradeTimeoutState,
        features: SignalFeatures,
        pips_gain: float,
        tick_rate: float,
        elapsed_sec: float,
    ) -> float:
        dir_mom = features.short_momentum_pips if state.side == "long" else -features.short_momentum_pips
        imbalance = features.momentum_pips if state.side == "long" else -features.momentum_pips
        spread_norm = _clamp((features.spread_pips - 0.1) / 0.6, 0.0, 1.2)
        tick_decay = state.events_used / max(state.event_budget_base, 1)
        time_ratio = elapsed_sec / max(state.timeout_sec_current, 0.6)
        decay = _clamp(max(tick_decay, time_ratio), 0.0, 1.5)
        momentum_norm = _clamp((dir_mom + 0.4) / 0.9, 0.0, 1.0)
        imbalance_norm = _clamp((imbalance + 0.35) / 0.9, 0.0, 1.0)
        latency_norm = _clamp((state.last_latency_ms - 90.0) / 480.0, 0.0, 1.0)
        pnl_norm = _clamp((pips_gain + 0.9) / 1.8, 0.0, 1.0)
        health = (
            0.5 * momentum_norm
            + 0.3 * imbalance_norm
            + 0.1 * pnl_norm
            - 0.4 * spread_norm
            - 0.3 * latency_norm
            - 0.2 * decay
        )
        return _clamp(health, -1.0, 1.0)

    def _hazard_check(
        self,
        state: TradeTimeoutState,
        features: SignalFeatures,
        tick_rate: float,
        latency_ms: float,
    ) -> tuple[float, float, float, bool]:
        dir_mom = features.short_momentum_pips if state.side == "long" else -features.short_momentum_pips
        imbalance = features.momentum_pips if state.side == "long" else -features.momentum_pips
        spread = features.spread_pips
        score = (
            config.HAZARD_INTERCEPT
            + config.HAZARD_MOMENTUM_COEF * dir_mom
            + config.HAZARD_IMBALANCE_COEF * imbalance
            - config.HAZARD_SPREAD_COEF * spread
            - config.HAZARD_LATENCY_COEF * latency_ms
        )
        p_tp = _sigmoid(score)
        p_sl = 1.0 - p_tp
        # cost係数: スプレッドとレイテンシが高いほど SL 先行リスクに重み付け
        spread_base = float(
            get_tuning_value(
                ("exit", "lowvol", "hazard_cost_spread_base"),
                config.HAZARD_COST_SPREAD_BASE,
            )
            or config.HAZARD_COST_SPREAD_BASE
        )
        latency_base = float(
            get_tuning_value(
                ("exit", "lowvol", "hazard_cost_latency_base_ms"),
                config.HAZARD_COST_LATENCY_BASE_MS,
            )
            or config.HAZARD_COST_LATENCY_BASE_MS
        )
        spread_base = max(0.05, spread_base)
        latency_base = max(50.0, latency_base)
        cost_k = 1.0 + spread / spread_base + latency_ms / latency_base + _clamp((10.0 - tick_rate) / 10.0, 0.0, 0.6)
        hazard_hit = p_tp < (p_sl * cost_k)
        return p_tp, p_sl, cost_k, hazard_hit

    def _adaptive_timeout_sec(
        self,
        *,
        tick_rate: float,
        spread: float,
        latency_ms: float,
        vola_ratio: float,
        ai_strength: float,
        extra_sec: float,
    ) -> float:
        base = 1.8 - 0.6 * (ai_strength - 0.7)
        tick_adj = 0.25 * (10.0 / max(tick_rate, 4.0))
        spread_adj = 0.35 * _clamp(spread / 0.3, 0.3, 2.0)
        latency_adj = 0.25 * _clamp(latency_ms / 180.0, 0.0, 2.0)
        vola_adj = 0.15 * (vola_ratio - 1.0)
        timeout = base + tick_adj + spread_adj + latency_adj + vola_adj + extra_sec
        tuned_max = float(
            get_tuning_value(
                ("exit", "lowvol", "upper_bound_max_sec"),
                config.TIMEOUT_ADAPTIVE_MAX_SEC,
            )
            or config.TIMEOUT_ADAPTIVE_MAX_SEC
        )
        tuned_max = max(config.TIMEOUT_ADAPTIVE_MIN_SEC, tuned_max)
        return _clamp(timeout, config.TIMEOUT_ADAPTIVE_MIN_SEC, tuned_max)

    def _classify_timeout(
        self,
        *,
        pips_gain: float,
        tick_rate: float,
        entry_slip: float,
        spread: float,
    ) -> str:
        if entry_slip >= config.TIMEOUT_SLIP_PIPS_THRESHOLD or spread >= config.TIMEOUT_SPREAD_SPIKE_PIPS:
            return "timeout_slip"
        if pips_gain <= -config.TIMEOUT_ADVERSE_PIPS_THRESHOLD:
            return "timeout_adverse"
        if abs(pips_gain) <= config.TIMEOUT_FLAT_PIPS_THRESHOLD and tick_rate <= config.TIMEOUT_FLAT_TICKRATE_THRESHOLD:
            return "timeout_flat"
        return "timeout_flat"
