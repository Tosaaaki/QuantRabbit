"""
execution.exit_manager
~~~~~~~~~~~~~~~~~~~~~~
注文のクローズ判定を担当。
• 逆方向シグナル or 指標の劣化を検知してクローズ指示を返す
• イベント時のポケット縮退もここでハンドル
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional, Tuple

from analysis.chart_story import ChartStorySnapshot

from analysis.ma_projection import MACrossProjection, compute_ma_projection
from utils.metrics_logger import log_metric
from advisors.exit_advisor import ExitHint
from workers.fast_scalp import config as fast_scalp_config


@dataclass
class ExitDecision:
    pocket: str
    units: int
    reason: str
    tag: str
    allow_reentry: bool = False


class ExitManager:
    def __init__(self, confidence_threshold: int = 70):
        self.confidence_threshold = confidence_threshold
        self._macro_signal_threshold = max(confidence_threshold + 10, 80)
        self._macro_trend_adx = 16
        self._macro_loss_buffer = 8.0
        self._macro_ma_gap = 3.0
        # Macro-specific stability controls
        self._macro_min_hold_minutes = 60.0  # insist on true H4-sized holding periods
        self._macro_hysteresis_pips = 12.0   # avoid reacting to M1 noise
        self._macro_trail_trigger_pips = 12.0
        self._macro_trail_offset = 0.0015
        # Micro-specific minimum hold
        self._micro_min_hold_minutes = 6.0
        self._reverse_confirmations = 3
        self._reverse_decay = timedelta(seconds=300)
        self._reverse_hits: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._range_macro_grace_minutes = 10.0
        self._range_macro_take_profit = 2.4
        self._range_macro_hold = 0.8
        self._range_macro_stop = 1.4
        self._pip = 0.01
        self._mfe_guard_base_default = 1.0
        self._mfe_guard_base = {"macro": 1.2, "micro": 1.0, "scalp": 0.85}
        self._mfe_guard_ratio = {"macro": 0.72, "micro": 0.6, "scalp": 0.55}
        self._mfe_sensitive_reasons = {
            "reverse_signal",
            "ma_cross",
            "ma_cross_imminent",
            "trend_reversal",
            "macro_trail_hit",
        }
        self._partial_eligible_reasons = {
            "reverse_signal",
            "ma_cross",
            "ma_cross_imminent",
            "trend_reversal",
            "macro_trail_hit",
        }
        self._force_exit_reasons = {
            "range_stop",
            "stop_loss_order",
            "macro_stop",
            "macro_stop_loss",
            "micro_momentum_stop",
            "event_lock",
        }
        self._min_partial_units = 600
        self._scalp_time_close_sec = 55.0
        self._scalp_time_target_pips = 0.95
        self._scalp_time_max_loss_pips = 4.0

    def _macro_trend_supports(
        self,
        direction: str,
        ma10: Optional[float],
        ma20: Optional[float],
        adx: float,
        slope_support: bool,
        cross_support: bool,
    ) -> bool:
        if ma10 is None or ma20 is None:
            return False
        if direction == "long":
            trend_ok = ma10 >= ma20
        else:
            trend_ok = ma10 <= ma20
        adx_ok = adx >= (self._macro_trend_adx - 1.5)
        momentum_ok = slope_support or cross_support
        return trend_ok and (adx_ok or momentum_ok)

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
        range_mode: bool = False,
        stage_state: Optional[Dict[str, Dict[str, int]]] = None,
        pocket_profiles: Optional[Dict[str, Dict[str, float]]] = None,
        now: Optional[datetime] = None,
        story: Optional[ChartStorySnapshot] = None,
        advisor_hints: Optional[Dict[str, ExitHint]] = None,
    ) -> List[ExitDecision]:
        current_time = self._ensure_utc(now)
        decisions: List[ExitDecision] = []
        projection_m1 = compute_ma_projection(fac_m1, timeframe_minutes=1.0)
        projection_h4 = compute_ma_projection(fac_h4, timeframe_minutes=240.0)
        m1_candles = fac_m1.get("candles") or []
        atr_m1 = self._safe_float(fac_m1.get("atr_pips"))
        atr_h4 = self._safe_float(fac_h4.get("atr_pips"))
        for pocket, info in open_positions.items():
            if pocket == "__net__":
                continue
            long_units = int(info.get("long_units", 0) or 0)
            short_units = int(info.get("short_units", 0) or 0)
            if long_units == 0:
                self._reset_reverse_counter(pocket, "long")
            if short_units == 0:
                self._reset_reverse_counter(pocket, "short")
            if long_units == 0 and short_units == 0:
                continue

            if pocket == "scalp_fast":
                decisions.extend(
                    self._evaluate_scalp_fast(
                        pocket=pocket,
                        open_info=info,
                        event_soon=event_soon,
                        now=current_time,
                        story=story,
                        range_mode=range_mode,
                    )
                )
                continue

            reverse_short = self._confirm_reverse_signal(
                self._strong_signal(signals, pocket, "OPEN_SHORT"),
                pocket,
                "long",
                current_time,
            )
            reverse_long = self._confirm_reverse_signal(
                self._strong_signal(signals, pocket, "OPEN_LONG"),
                pocket,
                "short",
                current_time,
            )

            pocket_fac = fac_h4 if pocket == "macro" else fac_m1
            rsi = pocket_fac.get("rsi", fac_m1.get("rsi", 50.0))
            ma10 = pocket_fac.get("ma10", 0.0)
            ma20 = pocket_fac.get("ma20", 0.0)
            adx = pocket_fac.get("adx", 0.0)
            ema_m1 = fac_m1.get("ema20")
            ema_h4 = fac_h4.get("ema20")
            close_price = fac_m1.get("close", 0.0)
            projection_primary = projection_h4 if pocket == "macro" else projection_m1
            projection_fast = projection_m1
            stage_long = ((stage_state or {}).get(pocket) or {}).get("long", 0)
            stage_short = ((stage_state or {}).get(pocket) or {}).get("short", 0)
            profile = (pocket_profiles or {}).get(pocket, {})

            if long_units > 0:
                decision = self._evaluate_long(
                    pocket,
                    info,
                    long_units,
                    reverse_short,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema_m1,
                    ema_h4 if pocket == "macro" else None,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    m1_candles,
                    story,
                    stage_level=stage_long,
                    pocket_profile=profile,
                    atr_primary=atr_h4 if pocket == "macro" else atr_m1,
                    atr_m1=atr_m1,
                    advisor_hints=advisor_hints,
                )
                if decision:
                    decisions.append(decision)

            if short_units > 0:
                decision = self._evaluate_short(
                    pocket,
                    info,
                    short_units,
                    reverse_long,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema_m1,
                    ema_h4 if pocket == "macro" else None,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    m1_candles,
                    story,
                    stage_level=stage_short,
                    pocket_profile=profile,
                    atr_primary=atr_h4 if pocket == "macro" else atr_m1,
                    atr_m1=atr_m1,
                    advisor_hints=advisor_hints,
                )
                if decision:
                    decisions.append(decision)

        return decisions

    def _strong_signal(
        self, signals: List[Dict], pocket: str, action: str
    ) -> Optional[Dict]:
        candidates = [
            s
            for s in signals
            if s.get("pocket") == pocket and s.get("action") == action
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s.get("confidence", 0))
        threshold = self.confidence_threshold
        if pocket == "macro":
            threshold = self._macro_signal_threshold
        if best.get("confidence", 0) >= threshold:
            return best
        return None

    def _evaluate_long(
        self,
        pocket: str,
        open_info: Dict,
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema_fast: float,
        ema_primary: Optional[float],
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        m1_candles: List[Dict],
        story: Optional[ChartStorySnapshot],
        *,
        stage_level: int = 0,
        pocket_profile: Optional[Dict[str, float]] = None,
        atr_primary: Optional[float] = None,
        atr_m1: Optional[float] = None,
        advisor_hints: Optional[Dict[str, ExitHint]] = None,
    ) -> Optional[ExitDecision]:
        allow_reentry = False
        reason = ""
        tag = f"{pocket}-long"
        avg_price = open_info.get("long_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (close_price - avg_price) / 0.01
        profile = pocket_profile or {}
        ema_ref = (
            ema_primary
            if pocket == "macro" and ema_primary is not None
            else ema_fast
        )
        hold_seconds: Optional[float] = None
        if pocket == "scalp":
            hold_seconds = self._trade_age_seconds(open_info, "short", now)
        hold_seconds: Optional[float] = None
        if pocket == "scalp":
            hold_seconds = self._trade_age_seconds(open_info, "long", now)

        advisor_hint = None
        if advisor_hints:
            advisor_hint = advisor_hints.get(f"{pocket}:long")
        if advisor_hint:
            advisor_decision = self._apply_advisor_hint(
                advisor_hint,
                pocket,
                "long",
                units,
                profit_pips,
                tag,
                story,
                range_mode,
                now,
            )
            if advisor_decision:
                return advisor_decision

        ma_gap_pips = 0.0
        if ma10 is not None and ma20 is not None:
            ma_gap_pips = abs(ma10 - ma20) / 0.01

        slope_support = (
            projection_fast is not None
            and projection_fast.gap_pips > 0.0
            and projection_fast.gap_slope_pips > 0.12
        )
        cross_support = (
            projection_primary is not None
            and projection_primary.gap_pips > 0.0
            and projection_primary.gap_slope_pips > 0.05
        )
        macd_cross_minutes = self._macd_cross_minutes(projection_fast, "long")

        if (
            reverse_signal
            and pocket == "macro"
            and profit_pips >= 4.0
            and close_price is not None
            and ema_ref is not None
            and close_price >= ema_ref + 0.002
        ):
            if (
                adx >= self._macro_trend_adx + 4
                or ma_gap_pips <= self._macro_ma_gap
                or slope_support
                or cross_support
            ):
                reverse_signal = None

        # Do not act on fresh macro trades unless conditions are clearly adverse
        if reverse_signal and pocket == "macro" and not range_mode:
            is_mature = self._has_mature_trade(
                open_info, "long", now, self._macro_min_hold_minutes
            )
            if not is_mature:
                early_exit_ok = (
                    profit_pips <= -self._macro_loss_buffer
                    or (
                        (ma10 is not None and ma20 is not None and ma10 < ma20)
                        and adx <= (self._macro_trend_adx - 2)
                        and ma_gap_pips >= (self._macro_ma_gap + 1.0)
                    )
                )
                if not early_exit_ok:
                    reverse_signal = None

        # Night/event guard for short-lived pockets
        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.6 < profit_pips < 1.6)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
                # Micro pocket: enforce a minimum hold before obeying reverse
                if pocket == "micro" and not range_mode:
                    if not self._has_mature_trade(open_info, "long", now, self._micro_min_hold_minutes):
                        return None
                if micro_guard or macro_guard:
                    return None
                reason = "reverse_signal"
                tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi >= 65:
            reason = "rsi_overbought"
        elif (
            pocket == "macro"
            and ma10 is not None
            and ma20 is not None
            and ma10 < ma20
            and adx <= self._macro_trend_adx
            and profit_pips <= -self._macro_loss_buffer
            and ma_gap_pips >= self._macro_ma_gap
        ):
            reason = "trend_reversal"
        elif pocket == "scalp":
            if hold_seconds is not None:
                if profit_pips <= -self._scalp_time_max_loss_pips:
                    reason = "scalp_time_stop"
                elif (
                    hold_seconds >= self._scalp_time_close_sec
                    and profit_pips < self._scalp_time_target_pips
                ):
                    reason = "scalp_time_exit"
                    allow_reentry = True
            if not reason and ema_fast is not None and close_price > ema_fast:
                reason = "scalp_momentum_flip"
        elif self._should_exit_for_cross(
            pocket,
            "long",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
        ):
            reason = "ma_cross_imminent"
        elif (
            pocket == "macro"
            and profit_pips >= self._macro_trail_trigger_pips
            and close_price is not None
            and ema_ref is not None
            and close_price <= ema_ref - self._macro_trail_offset
        ):
            reason = "macro_trail_hit"
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if pocket == "macro":
                if self._has_mature_trade(open_info, "long", now, self._range_macro_grace_minutes):
                    return None
                tp = self._range_macro_take_profit
                hold = self._range_macro_hold
                stop = self._range_macro_stop
                if profit_pips >= tp:
                    reason = "range_take_profit"
                elif profit_pips > hold:
                    return None
                elif profit_pips <= -stop:
                    reason = "range_stop"
            else:
                if profit_pips >= 1.6:
                    reason = "range_take_profit"
                elif profit_pips > 0.4:
                    return None
                elif profit_pips <= -1.0:
                    reason = "range_stop"

        # MFE-based patience: if we've achieved decent favorable excursion,
        # avoid exiting on a mild pullback unless strong invalidation.
        if reason in self._mfe_sensitive_reasons and not range_mode:
            guard_threshold, guard_ratio = self._get_mfe_guard(pocket, atr_primary)
            max_mfe = self._max_mfe_for_side(open_info, "long", m1_candles, now)
            if max_mfe is not None and max_mfe >= guard_threshold:
                retrace = max(0.0, max_mfe - max(0.0, profit_pips))
                if retrace <= guard_ratio * max_mfe and profit_pips > -self._macro_loss_buffer:
                    return None

        if reason == "trend_reversal":
            if not self._validate_trend_reversal(
                pocket,
                "long",
                story,
                close_price,
                m1_candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
            ):
                reason = ""
        elif reason == "macro_trail_hit":
            if not self._validate_trend_reversal(
                pocket,
                "long",
                story,
                close_price,
                m1_candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
                bias_only=True,
            ):
                reason = ""

        if (
            pocket == "macro"
            and reason
            and not range_mode
            and profit_pips > -self._macro_loss_buffer
        ):
            trend_supports = self._macro_trend_supports(
                "long", ma10, ma20, adx, slope_support, cross_support
            )
            mature = self._has_mature_trade(
                open_info, "long", now, self._macro_min_hold_minutes
            )
            if trend_supports and not mature and reason in {
                "reverse_signal",
                "trend_reversal",
                "ma_cross",
                "ma_cross_imminent",
            }:
                return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason:
            if not self._pattern_supports_exit(
                story,
                pocket,
                "long",
                reason,
                profit_pips,
            ):
                return None
            if not self._story_allows_exit(
                story,
                pocket,
                "long",
                reason,
                profit_pips,
                now,
                range_mode=range_mode,
            ):
                return None
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        close_units = self._compute_exit_units(
            pocket,
            "long",
            reason,
            units,
            stage_level,
            profile,
            range_mode=range_mode,
            profit_pips=profit_pips,
        )
        if close_units <= 0:
            return None

        self._record_exit_metric(
            pocket,
            "long",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

        age_seconds = self._trade_age_seconds(open_info, "long", now)
        hold_minutes = round(age_seconds / 60.0, 2) if age_seconds is not None else None
        logging.info(
            "[EXIT] pocket=%s side=long reason=%s profit=%.2fp hold=%smin close_units=%s range=%s stage=%d",
            pocket,
            reason,
            profit_pips,
            f"{hold_minutes:.2f}" if hold_minutes is not None else "n/a",
            close_units,
            range_mode,
            stage_level,
        )

        return ExitDecision(
            pocket=pocket,
            units=-abs(close_units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    def _evaluate_short(
        self,
        pocket: str,
        open_info: Dict,
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema_fast: float,
        ema_primary: Optional[float],
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        m1_candles: List[Dict],
        story: Optional[ChartStorySnapshot],
        *,
        stage_level: int = 0,
        pocket_profile: Optional[Dict[str, float]] = None,
        atr_primary: Optional[float] = None,
        atr_m1: Optional[float] = None,
        advisor_hints: Optional[Dict[str, ExitHint]] = None,
    ) -> Optional[ExitDecision]:
        allow_reentry = False
        reason = ""
        tag = f"{pocket}-short"
        avg_price = open_info.get("short_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (avg_price - close_price) / 0.01
        profile = pocket_profile or {}
        ema_ref = (
            ema_primary
            if pocket == "macro" and ema_primary is not None
            else ema_fast
        )

        advisor_hint = None
        if advisor_hints:
            advisor_hint = advisor_hints.get(f"{pocket}:short")
        if advisor_hint:
            advisor_decision = self._apply_advisor_hint(
                advisor_hint,
                pocket,
                "short",
                units,
                profit_pips,
                tag,
                story,
                range_mode,
                now,
            )
            if advisor_decision:
                return advisor_decision

        ma_gap_pips = 0.0
        if ma10 is not None and ma20 is not None:
            ma_gap_pips = abs(ma10 - ma20) / 0.01

        slope_support = (
            projection_fast is not None
            and projection_fast.gap_pips < 0.0
            and projection_fast.gap_slope_pips < -0.12
        )
        cross_support = (
            projection_primary is not None
            and projection_primary.gap_pips < 0.0
            and projection_primary.gap_slope_pips < -0.05
        )
        macd_cross_minutes = self._macd_cross_minutes(projection_fast, "short")

        if (
            reverse_signal
            and pocket == "macro"
            and profit_pips >= 4.0
            and close_price is not None
            and ema_ref is not None
            and close_price <= ema_ref - 0.002
        ):
            if (
                adx >= self._macro_trend_adx + 4
                or ma_gap_pips <= self._macro_ma_gap
                or slope_support
                or cross_support
            ):
                reverse_signal = None

        if (
            pocket == "micro"
            and profit_pips <= -4.0
            and close_price is not None
            and ema_fast is not None
            and (
                close_price >= ema_fast + 0.0015
                or rsi >= 55
                or (ma10 is not None and ma20 is not None and ma10 > ma20)
            )
        ):
            reason = "micro_momentum_stop"
        elif event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.6 < profit_pips < 1.6)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
                # Micro pocket: enforce a minimum hold before obeying reverse
                if pocket == "micro" and not range_mode:
                    if not self._has_mature_trade(open_info, "short", now, self._micro_min_hold_minutes):
                        return None
                if micro_guard or macro_guard:
                    return None
                reason = "reverse_signal"
                tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi <= 35:
            reason = "rsi_oversold"
        elif (
            pocket == "macro"
            and ma10 is not None
            and ma20 is not None
            and ma10 > ma20
            and adx <= self._macro_trend_adx
            and profit_pips <= -self._macro_loss_buffer
            and ma_gap_pips >= self._macro_ma_gap
        ):
            reason = "trend_reversal"
        elif pocket == "scalp":
            if hold_seconds is not None:
                if profit_pips <= -self._scalp_time_max_loss_pips:
                    reason = "scalp_time_stop"
                elif (
                    hold_seconds >= self._scalp_time_close_sec
                    and profit_pips < self._scalp_time_target_pips
                ):
                    reason = "scalp_time_exit"
                    allow_reentry = True
            if not reason and ema_fast is not None and close_price < ema_fast:
                reason = "scalp_momentum_flip"
        elif self._should_exit_for_cross(
            pocket,
            "short",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
        ):
            reason = "ma_cross_imminent"
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if pocket == "macro":
                if self._has_mature_trade(open_info, "short", now, self._range_macro_grace_minutes):
                    return None
                tp = self._range_macro_take_profit
                hold = self._range_macro_hold
                stop = self._range_macro_stop
                if profit_pips >= tp:
                    reason = "range_take_profit"
                elif profit_pips > hold:
                    return None
                elif profit_pips <= -stop:
                    reason = "range_stop"
            else:
                if profit_pips >= 1.6:
                    reason = "range_take_profit"
                elif profit_pips > 0.4:
                    return None
                elif profit_pips <= -1.0:
                    reason = "range_stop"

        if reason in self._mfe_sensitive_reasons and not range_mode:
            guard_threshold, guard_ratio = self._get_mfe_guard(pocket, atr_primary)
            max_mfe = self._max_mfe_for_side(open_info, "short", m1_candles, now)
            if max_mfe is not None and max_mfe >= guard_threshold:
                retrace = max(0.0, max_mfe - max(0.0, -profit_pips))
                if retrace <= guard_ratio * max_mfe and profit_pips < self._macro_loss_buffer:
                    return None

        if reason == "trend_reversal":
            if not self._validate_trend_reversal(
                pocket,
                "short",
                story,
                close_price,
                m1_candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
            ):
                reason = ""
        elif reason == "macro_trail_hit":
            if not self._validate_trend_reversal(
                pocket,
                "short",
                story,
                close_price,
                m1_candles,
                atr_primary=atr_primary,
                atr_m1=atr_m1,
                bias_only=True,
            ):
                reason = ""

        if (
            pocket == "macro"
            and reason
            and not range_mode
            and profit_pips > -self._macro_loss_buffer
        ):
            trend_supports = self._macro_trend_supports(
                "short", ma10, ma20, adx, slope_support, cross_support
            )
            mature = self._has_mature_trade(
                open_info, "short", now, self._macro_min_hold_minutes
            )
            if trend_supports and not mature and reason in {
                "reverse_signal",
                "trend_reversal",
                "ma_cross",
                "ma_cross_imminent",
            }:
                return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason:
            if not self._pattern_supports_exit(
                story,
                pocket,
                "short",
                reason,
                profit_pips,
            ):
                return None
            if not self._story_allows_exit(
                story,
                pocket,
                "short",
                reason,
                profit_pips,
                now,
                range_mode=range_mode,
            ):
                return None
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        close_units = self._compute_exit_units(
            pocket,
            "short",
            reason,
            units,
            stage_level,
            profile,
            range_mode=range_mode,
            profit_pips=profit_pips,
        )
        if close_units <= 0:
            return None

        self._record_exit_metric(
            pocket,
            "short",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

        age_seconds = self._trade_age_seconds(open_info, "short", now)
        hold_minutes = round(age_seconds / 60.0, 2) if age_seconds is not None else None
        logging.info(
            "[EXIT] pocket=%s side=short reason=%s profit=%.2fp hold=%smin close_units=%s range=%s stage=%d",
            pocket,
            reason,
            profit_pips,
            f"{hold_minutes:.2f}" if hold_minutes is not None else "n/a",
            close_units,
            range_mode,
            stage_level,
        )

        return ExitDecision(
            pocket=pocket,
            units=abs(close_units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    def _evaluate_scalp_fast(
        self,
        pocket: str,
        open_info: Dict,
        event_soon: bool,
        now: datetime,
        story: Optional[ChartStorySnapshot],
        range_mode: bool,
    ) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        trades = open_info.get("open_trades") or []
        for tr in trades:
            units = abs(int(tr.get("units", 0) or 0))
            if units == 0:
                continue
            side = tr.get("side", "long")
            profit_pips = float(tr.get("unrealized_pl_pips") or 0.0)
            entry_meta = tr.get("entry_thesis") or {}
            pattern_score = entry_meta.get("pattern_score")
            try:
                pattern_score = float(pattern_score)
            except (TypeError, ValueError):
                pattern_score = None
            pattern_tag = entry_meta.get("pattern_tag")
            hold_seconds = None
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is not None:
                hold_seconds = (now - opened_at).total_seconds()

            target = 1.6
            timeout = 40.0
            max_loss = 1.8
            if pattern_score is not None:
                if pattern_score >= 0.8:
                    target = 2.4
                    timeout = 60.0
                    max_loss = 2.4
                elif pattern_score >= fast_scalp_config.PATTERN_MIN_PROB:
                    target = 1.8
                    timeout = 50.0
                else:
                    target = 1.3
                    timeout = 35.0
                    max_loss = 1.4

            reason = ""
            allow_reentry = False

            if event_soon:
                reason = "scalp_fast_event_exit"
                allow_reentry = True
            elif profit_pips >= target and (hold_seconds or 0.0) >= 8.0:
                reason = "scalp_fast_target_hit"
            elif profit_pips <= -max_loss:
                reason = "scalp_fast_stop"
            elif hold_seconds is not None and hold_seconds >= timeout and profit_pips < target:
                reason = "scalp_fast_timeout"
                allow_reentry = True
            elif (
                pattern_score is not None
                and pattern_score < fast_scalp_config.PATTERN_MIN_PROB
                and profit_pips < 0.3
                and (hold_seconds or 0.0) >= 12.0
            ):
                reason = "scalp_fast_pattern_veto"
                allow_reentry = True
            elif (
                pattern_score is not None
                and pattern_score >= 0.7
                and profit_pips >= 1.0
                and (hold_seconds or 0.0) >= 12.0
            ):
                reason = "scalp_fast_secure"
            elif (
                pattern_tag in {"flat_chop", "narrow_range", "broad_range_stall"}
                and (hold_seconds or 0.0) >= 20.0
                and profit_pips < 0.6
            ):
                reason = "scalp_fast_chop_exit"
                allow_reentry = True

            if not reason and range_mode:
                if profit_pips >= 1.4:
                    reason = "scalp_fast_range_tp"
                elif profit_pips <= -1.0:
                    reason = "scalp_fast_range_stop"

            if not reason:
                continue

            side_units = -units if side == "long" else units
            self._record_exit_metric(
                pocket,
                side,
                reason,
                profit_pips,
                story,
                range_mode,
                now,
            )
            logging.info(
                "[EXIT] pocket=%s side=%s reason=%s profit=%.2fp pattern=%.3f tag=%s",
                pocket,
                side,
                reason,
                profit_pips,
                pattern_score if pattern_score is not None else float("nan"),
                pattern_tag or "",
            )
            decisions.append(
                ExitDecision(
                    pocket=pocket,
                    units=side_units,
                    reason=reason,
                    tag=f"{pocket}-{side}",
                    allow_reentry=allow_reentry,
                )
            )
        return decisions


    def _should_exit_for_cross(
        self,
        pocket: str,
        side: str,
        open_info: Dict,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        profit_pips: float,
        now: datetime,
        macd_cross_minutes: Optional[float],
    ) -> bool:
        projection = projection_fast or projection_primary
        if projection is None:
            return False

        gap = projection.gap_pips
        if side == "long" and gap < 0.0:
            return True
        if side == "short" and gap > 0.0:
            return True

        candidates: List[float] = []
        if projection.projected_cross_minutes is not None:
            candidates.append(projection.projected_cross_minutes)
        if macd_cross_minutes is not None:
            candidates.append(macd_cross_minutes)
        if not candidates:
            return False
        cross_minutes = min(candidates)

        slope_source = projection_fast or projection_primary
        slope = slope_source.gap_slope_pips if slope_source else 0.0
        if macd_cross_minutes is None:
            if side == "long" and slope >= 0.0:
                return False
            if side == "short" and slope <= 0.0:
                return False

        threshold = 3.5
        mature_macro_trade = False
        if pocket == "macro":
            threshold = 10.0
            mature_macro_trade = self._has_mature_trade(
                open_info, side, now, self._macro_min_hold_minutes
            )
            if not mature_macro_trade:
                threshold = 5.0
        elif pocket == "scalp":
            threshold = 2.2

        if cross_minutes > threshold:
            return False

        if pocket == "macro":
            if not mature_macro_trade and profit_pips > -self._macro_loss_buffer:
                return False
            if profit_pips <= -self._macro_loss_buffer:
                return True
            if cross_minutes <= threshold / 2.0:
                return True
            if macd_cross_minutes is not None and macd_cross_minutes <= threshold / 2.0:
                return True
            slope_check = slope_source.gap_slope_pips if slope_source else 0.0
            if profit_pips < 2.0 and cross_minutes <= threshold:
                return True
            if slope_check < -0.08 and cross_minutes <= threshold:
                return True
            return False

        if profit_pips >= 0.8:
            return True
        if cross_minutes <= threshold / 2.0:
            return True
        if macd_cross_minutes is not None and macd_cross_minutes <= threshold / 2.0:
            return True
        return False

    @staticmethod
    def _macd_cross_minutes(
        projection: Optional[MACrossProjection],
        side: str,
    ) -> Optional[float]:
        if (
            projection is None
            or projection.macd_pips is None
            or projection.macd_slope_pips is None
        ):
            return None
        macd = projection.macd_pips
        slope = projection.macd_slope_pips
        if side == "long":
            if macd <= 0.0 and slope <= 0.0:
                return 0.0
            if macd > 0.0 and slope < 0.0 and projection.macd_cross_minutes is not None:
                return projection.macd_cross_minutes
        else:
            if macd >= 0.0 and slope >= 0.0:
                return 0.0
            if macd < 0.0 and slope > 0.0 and projection.macd_cross_minutes is not None:
                return projection.macd_cross_minutes
        return None

    @staticmethod
    def _parse_candle_time(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        t = raw.strip()
        try:
            if t.endswith("Z"):
                t = t[:-1] + "+00:00"
            if "." in t and "+" not in t:
                head, frac = t.split(".", 1)
                frac = "".join(ch for ch in frac if ch.isdigit())[:6]
                t = f"{head}.{frac}+00:00"
            elif "+" not in t:
                t = f"{t}+00:00"
            return datetime.fromisoformat(t)
        except Exception:
            try:
                base = t.split(".", 1)[0].rstrip("Z") + "+00:00"
                return datetime.fromisoformat(base)
            except Exception:
                return None

    def _max_mfe_for_side(
        self,
        open_info: Dict,
        side: str,
        m1_candles: List[Dict],
        now: datetime,
    ) -> Optional[float]:
        trades = [
            tr for tr in (open_info.get("open_trades") or []) if tr.get("side") == side
        ]
        if not trades or not m1_candles:
            return None
        # Prepare candle tuples (ts, h, l)
        candles: List[Tuple[datetime, float, float]] = []
        for c in m1_candles:
            ts = self._parse_candle_time(c.get("timestamp"))
            if not ts:
                continue
            try:
                h = float(c.get("high"))
                l = float(c.get("low"))
            except Exception:
                continue
            candles.append((ts, h, l))
        if not candles:
            return None
        max_mfe = 0.0
        for tr in trades:
            ep = tr.get("price")
            ot = self._parse_open_time(tr.get("open_time"))
            if ep is None or ot is None:
                continue
            for ts, h, l in candles:
                if ts < ot or ts > now:
                    continue
                if side == "long":
                    fav = (h - ep) / 0.01
                else:
                    fav = (ep - l) / 0.01
                if fav > max_mfe:
                    max_mfe = fav
        return round(max_mfe, 2)

    def _confirm_reverse_signal(
        self,
        signal: Optional[Dict],
        pocket: str,
        direction: str,
        now: datetime,
    ) -> Optional[Dict]:
        key = (pocket, direction)
        state = self._reverse_hits.get(key)
        if signal:
            if state:
                ts = state.get("ts")
                if isinstance(ts, datetime) and now - ts > self._reverse_decay:
                    state = None
            count = 0
            if state:
                count = int(state.get("count", 0) or 0)
            count += 1
            self._reverse_hits[key] = {"count": count, "ts": now}
            needed = self._reverse_confirmations + (1 if pocket == "macro" else 0)
            if count >= needed:
                return signal
            return None
        if state:
            ts = state.get("ts")
            if isinstance(ts, datetime) and now - ts > self._reverse_decay:
                self._reverse_hits.pop(key, None)
            else:
                self._reverse_hits[key] = {"count": 0, "ts": now}
        return None

    def _story_allows_exit(
        self,
        story: Optional[ChartStorySnapshot],
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
        now: datetime,
        *,
        range_mode: bool,
    ) -> bool:
        if story is None:
            return True
        if reason in {"range_stop", "stop_loss_order"}:
            return True

        trend = self._story_trend(story, pocket)
        supportive = False
        if side == "long" and trend == "up":
            supportive = True
        if side == "short" and trend == "down":
            supportive = True

        if not supportive:
            return True

        if reason in {"reverse_signal", "ma_cross_imminent", "ma_cross"}:
            if profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[STORY] sustain exit defer pocket=%s side=%s reason=%s trend=%s profit=%.2f",
                    pocket,
                    side,
                    reason,
                    trend,
                    profit_pips,
                )
                log_metric(
                    "exit_story_blocked",
                    float(profit_pips),
                    tags={
                        "pocket": pocket,
                        "side": side,
                        "reason": reason,
                        "trend": trend or "",
                        "volatility": getattr(story, "volatility_state", None) or "",
                        "range_mode": str(range_mode),
                    },
                    ts=now,
                )
                return False
        return True

    def _pattern_supports_exit(
        self,
        story: Optional[ChartStorySnapshot],
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
    ) -> bool:
        if story is None:
            return True
        patterns = getattr(story, "pattern_summary", None) or {}
        if not patterns or pocket != "macro":
            return True

        n_wave = patterns.get("n_wave") or {}
        bias = n_wave.get("bias")
        confidence = float(n_wave.get("confidence", 0.0) or 0.0)
        # Reasons that are soft / pattern-driven for macro exits
        pattern_reasons = {
            "reverse_signal",
            "ma_cross",
            "ma_cross_imminent",
            "trend_reversal",
            "macro_trail_hit",
        }
        if reason not in pattern_reasons:
            return True

        if side == "long":
            if bias == "up" and confidence >= 0.55 and profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[PATTERN] veto macro exit: bias=%s conf=%.2f profit=%.2f",
                    bias,
                    confidence,
                    profit_pips,
                )
                return False
            if bias == "down" and confidence >= 0.5:
                return True
        else:
            if bias == "down" and confidence >= 0.55 and profit_pips > -self._macro_loss_buffer:
                logging.info(
                    "[PATTERN] veto macro exit: bias=%s conf=%.2f profit=%.2f",
                    bias,
                    confidence,
                    profit_pips,
                )
                return False
            if bias == "up" and confidence >= 0.5:
                return True
        return True

    @staticmethod
    def _story_trend(
        story: Optional[ChartStorySnapshot],
        pocket: str,
    ) -> Optional[str]:
        if story is None:
            return None
        if pocket == "macro":
            return story.macro_trend
        if pocket == "micro":
            return story.micro_trend
        return story.higher_trend

    def _get_mfe_guard(
        self,
        pocket: str,
        atr_primary: Optional[float],
    ) -> Tuple[float, float]:
        base = self._mfe_guard_base.get(pocket, self._mfe_guard_base_default)
        atr = self._safe_float(atr_primary)
        if atr > 0.0:
            if pocket == "macro":
                base = max(base, min(base + 0.9, 0.9 + atr * 0.28))
            elif pocket == "micro":
                base = max(base, min(base + 0.6, 0.7 + atr * 0.22))
            elif pocket == "scalp":
                base = max(base, min(base + 0.4, 0.6 + atr * 0.18))
        ratio = self._mfe_guard_ratio.get(pocket, 0.6)
        return round(base, 2), ratio

    def _record_exit_metric(
        self,
        pocket: str,
        side: str,
        reason: str,
        profit_pips: float,
        story: Optional[ChartStorySnapshot],
        range_mode: bool,
        now: datetime,
    ) -> None:
        trend = self._story_trend(story, pocket) if story else None
        volatility = story.volatility_state if story else None
        summary_state = None
        if story and story.summary:
            summary_state = story.summary.get("H1")
        tags = {
            "pocket": pocket,
            "side": side,
            "reason": reason,
            "trend": trend or "",
            "volatility": volatility or "",
            "summary_h1": summary_state or "",
            "range_mode": str(range_mode),
        }
        log_metric(
            "exit_decision",
            float(profit_pips),
            tags=tags,
            ts=now,
        )
        log_metric(
            "exit_decision_count",
            1.0,
            tags=tags,
            ts=now,
        )

    def _apply_advisor_hint(
        self,
        hint: ExitHint,
        pocket: str,
        side: str,
        units: int,
        profit_pips: float,
        tag: str,
        story: Optional[ChartStorySnapshot],
        range_mode: bool,
        now: datetime,
    ) -> Optional[ExitDecision]:
        reason = None
        if hint.max_drawdown_pips is not None and profit_pips <= -abs(hint.max_drawdown_pips):
            reason = "advisor_drawdown"
        elif hint.min_takeprofit_pips is not None and profit_pips >= abs(hint.min_takeprofit_pips):
            reason = "advisor_takeprofit"
        if not reason:
            return None
        units_to_close = abs(units)
        signed_units = -units_to_close if side == "long" else units_to_close
        final_reason = reason
        if hint.reason:
            final_reason = f"{reason}:{hint.reason}"
        allow_reentry = hint.confidence >= 0.7
        self._record_exit_metric(
            pocket,
            side,
            final_reason,
            profit_pips,
            story,
            range_mode,
            now,
        )
        log_metric(
            "exit_advisor_trigger",
            float(profit_pips),
            tags={
                "pocket": pocket,
                "side": side,
                "reason": reason,
                "model": hint.model_used or "unknown",
            },
            ts=now,
        )
        return ExitDecision(
            pocket=pocket,
            units=signed_units,
            reason=final_reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    def _validate_trend_reversal(
        self,
        pocket: str,
        side: str,
        story: Optional[ChartStorySnapshot],
        close_price: Optional[float],
        m1_candles: List[Dict],
        *,
        atr_primary: Optional[float],
        atr_m1: Optional[float],
        bias_only: bool = False,
    ) -> bool:
        if close_price is None:
            return False
        trend = self._story_trend(story, pocket)
        higher = getattr(story, "higher_trend", None) if story else None
        structure_bias = getattr(story, "structure_bias", 0.0) if story else 0.0
        if side == "long":
            if trend == "down" or higher == "down" or structure_bias <= -4.0:
                return True
        else:
            if trend == "up" or higher == "up" or structure_bias >= 4.0:
                return True
        if bias_only:
            return False
        atr = atr_primary or atr_m1 or 0.0
        return self._is_structural_break(side, close_price, m1_candles, atr_pips=atr)

    def _is_structural_break(
        self,
        side: str,
        close_price: float,
        candles: List[Dict],
        *,
        atr_pips: float = 0.0,
        lookback: int = 12,
    ) -> bool:
        if not candles or len(candles) < lookback + 2:
            return False
        lows: List[float] = []
        highs: List[float] = []
        for c in candles[-(lookback + 2) : -1]:
            try:
                lows.append(float(c.get("low")))
                highs.append(float(c.get("high")))
            except (TypeError, ValueError):
                continue
        if not lows or not highs:
            return False
        atr_buffer = max(0.15, (atr_pips or 0.0) * 0.28)
        buffer_price = atr_buffer * self._pip
        if side == "long":
            swing_low = min(lows)
            return close_price <= swing_low - buffer_price
        swing_high = max(highs)
        return close_price >= swing_high + buffer_price

    def _compute_exit_units(
        self,
        pocket: str,
        side: str,
        reason: str,
        total_units: int,
        stage_level: int,
        pocket_profile: Dict[str, float],
        *,
        range_mode: bool,
        profit_pips: Optional[float] = None,
    ) -> int:
        base = abs(total_units)
        if base == 0:
            return 0
        if base <= self._min_partial_units:
            return base
        # 負けているポジションでは段階的クローズを避け、即時にまとめて縮小/クローズする
        if profit_pips is not None and profit_pips <= 0.0:
            return base
        if reason not in self._partial_eligible_reasons:
            return base
        if reason in self._force_exit_reasons:
            return base
        fraction = 0.7
        if reason == "trend_reversal":
            fraction = 0.6
        elif reason in {"reverse_signal", "ma_cross_imminent", "ma_cross"}:
            fraction = 0.55
        if stage_level >= 4:
            fraction *= 0.6
        elif stage_level == 3:
            fraction *= 0.7
        elif stage_level == 2:
            fraction *= 0.8
        win_rate = pocket_profile.get("win_rate", 0.0)
        avg_loss = pocket_profile.get("avg_loss_pips", 0.0)
        if win_rate >= 0.58:
            fraction *= 0.85
        if avg_loss and avg_loss <= 3.5:
            fraction *= 0.85
        if range_mode:
            fraction = min(0.85, fraction * 1.05)
        fraction = max(0.25, min(0.75, fraction))
        units_to_close = int(round(base * fraction))
        if units_to_close <= 0:
            units_to_close = 1
        preserve_floor = max(1, int(round(base * 0.25)))
        max_close = max(1, base - preserve_floor)
        if units_to_close >= base:
            units_to_close = max_close
        elif units_to_close > max_close:
            units_to_close = max_close
        if units_to_close <= 0:
            units_to_close = max_close
        return min(base, units_to_close)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _reset_reverse_counter(self, pocket: str, direction: str) -> None:
        self._reverse_hits.pop((pocket, direction), None)

    def _has_mature_trade(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
        threshold_minutes: float,
    ) -> bool:
        trades = open_info.get("open_trades") or []
        for tr in trades:
            if tr.get("side") != side:
                continue
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is None:
                continue
            age_minutes = (now - opened_at).total_seconds() / 60.0
            if age_minutes >= threshold_minutes:
                return True
        return False

    def _trade_age_seconds(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
    ) -> Optional[float]:
        trades = open_info.get("open_trades") or []
        ages: list[float] = []
        for tr in trades:
            if tr.get("side") != side:
                continue
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is None:
                continue
            ages.append((now - opened_at).total_seconds())
        if not ages:
            return None
        return max(ages)

    @staticmethod
    def _parse_open_time(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            if "." in raw:
                head, frac = raw.split(".", 1)
                frac_digits = "".join(ch for ch in frac if ch.isdigit())
                if len(frac_digits) > 6:
                    frac_digits = frac_digits[:6]
                tz_part = ""
                if "+" in raw:
                    tz_part = raw[raw.rfind("+") :]
                if not tz_part:
                    tz_part = "+00:00"
                raw = f"{head}.{frac_digits}{tz_part}"
            elif "+" not in raw:
                raw = f"{raw}+00:00"
            dt = datetime.fromisoformat(raw)
            return dt.astimezone(timezone.utc)
        except ValueError:
            try:
                trimmed = raw
                if "." in trimmed:
                    trimmed = trimmed.split(".", 1)[0]
                if not trimmed.endswith("+00:00"):
                    trimmed = trimmed.rstrip("Z") + "+00:00"
                dt = datetime.fromisoformat(trimmed)
                return dt.astimezone(timezone.utc)
            except ValueError:
                return None

    @staticmethod
    def _ensure_utc(candidate: Optional[datetime]) -> datetime:
        if candidate is None:
            return datetime.now(timezone.utc)
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)
