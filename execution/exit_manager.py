"""
execution.exit_manager
~~~~~~~~~~~~~~~~~~~~~~
注文のクローズ判定を担当。
• 逆方向シグナル or 指標の劣化を検知してクローズ指示を返す
• イベント時のポケット縮退もここでハンドル
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import logging

from analysis.ma_projection import MACrossProjection, compute_ma_projection
from analysis.mtf_utils import resample_candles_from_m1
from utils.metrics_logger import log_metric

if TYPE_CHECKING:
    from execution.stage_tracker import StageTracker


def _in_jst_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    """Return True when UTC time falls within the specified JST window."""
    start = start_hour % 24
    end = end_hour % 24
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    jst = current + timedelta(hours=9)
    hour = jst.hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


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
        self._macro_loss_buffer = 1.10
        self._macro_ma_gap = 3.0
        # Macro-specific stability controls
        self._macro_min_hold_minutes = 6.0  # hold a bit longer to reduce churn
        self._macro_hysteresis_pips = 1.6   # wider deadband; avoid near-flat exits
        # Pattern-aware retest handling (macro only)
        self._macro_retest_band_base = 1.2  # pips around fast MA treated as retest zone
        self._macro_retest_m5_slope = 0.06  # min M5 slope (pips/bar) aligning with position
        self._macro_retest_m10_slope = 0.04 # min M10 slope (pips/bar)
        self._macro_struct_cushion = 0.22   # ATR fraction for pivot kill-line cushion
        # Scalp-specific stability controls
        self._scalp_loss_guard = 1.15
        self._scalp_take_profit = 2.4
        self._reverse_confirmations = 2
        self._reverse_decay = timedelta(seconds=180)
        self._reverse_hits: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._range_macro_grace_minutes = 8.0
        # Micro-specific stability controls
        # Guard against premature exits on noisy micro trades: enforce longer holds and wider grace
        self._micro_min_hold_seconds = float(os.getenv("EXIT_MICRO_MIN_HOLD_SEC", "90"))
        self._micro_loss_grace_pips = float(os.getenv("EXIT_MICRO_GUARD_LOSS_PIPS", "2.5"))
        self._micro_loss_hold_seconds = float(os.getenv("EXIT_MICRO_LOSS_HOLD_SEC", "90"))
        self._micro_profit_hard = float(os.getenv("EXIT_MICRO_PROFIT_TAKE_PIPS", "1.60"))
        self._micro_profit_soft = float(os.getenv("EXIT_MICRO_PROFIT_SOFT_PIPS", "1.00"))
        self._micro_profit_rsi_release_long = float(os.getenv("EXIT_MICRO_PROFIT_RSI_LONG", "53"))
        self._micro_profit_rsi_release_short = float(os.getenv("EXIT_MICRO_PROFIT_RSI_SHORT", "47"))
        self._micro_profit_ema_buffer = float(os.getenv("EXIT_MICRO_PROFIT_EMA_BUFFER", "0.0005"))
        self._micro_profit_slope_min = float(os.getenv("EXIT_MICRO_PROFIT_SLOPE_MIN", "0.05"))
        # Scalp pocket guard rails
        self._scalp_min_hold_seconds = float(os.getenv("EXIT_SCALP_MIN_HOLD_SEC", "45"))
        self._scalp_loss_grace_pips = float(os.getenv("EXIT_SCALP_GUARD_LOSS_PIPS", "2.0"))
        # TrendMA / volatility-specific garde rails
        self._trendma_partial_fraction = float(os.getenv("EXIT_TRENDMA_PARTIAL_FRACTION", "0.5"))
        self._trendma_partial_profit_cap = float(os.getenv("EXIT_TRENDMA_PARTIAL_PROFIT_CAP", "3.4"))
        self._vol_partial_atr_min = float(os.getenv("EXIT_VOL_PARTIAL_ATR_MIN", "1.5"))
        self._vol_partial_atr_max = float(os.getenv("EXIT_VOL_PARTIAL_ATR_MAX", "2.6"))
        self._vol_partial_fraction = float(os.getenv("EXIT_VOL_PARTIAL_FRACTION", "0.66"))
        self._vol_partial_profit_floor = float(os.getenv("EXIT_VOL_PARTIAL_PROFIT_FLOOR", "2.5"))
        self._vol_partial_profit_cap = float(os.getenv("EXIT_VOL_PARTIAL_PROFIT_CAP", "3.0"))
        self._vol_ema_release_gap = float(os.getenv("EXIT_VOL_EMA_RELEASE_GAP", "1.0"))
        self._profit_snatch_min = float(os.getenv("EXIT_SNATCH_MIN_PROFIT_PIPS", "0.3"))
        self._profit_snatch_max = float(os.getenv("EXIT_SNATCH_MAX_PROFIT_PIPS", "0.8"))
        self._profit_snatch_hold = float(os.getenv("EXIT_SNATCH_MIN_HOLD_SEC", "70"))
        self._profit_snatch_atr_min = float(os.getenv("EXIT_SNATCH_ATR_MIN", "1.0"))
        self._profit_snatch_vol_min = float(os.getenv("EXIT_SNATCH_VOL5M_MIN", "0.8"))
        self._profit_snatch_jst_start = int(os.getenv("EXIT_SNATCH_JST_START", "0")) % 24
        self._profit_snatch_jst_end = int(os.getenv("EXIT_SNATCH_JST_END", "6")) % 24
        self._loss_guard_atr_trigger = float(os.getenv("EXIT_LOSS_GUARD_ATR_TRIGGER", "2.0"))
        self._loss_guard_vol_trigger = float(os.getenv("EXIT_LOSS_GUARD_VOL_TRIGGER", "1.5"))
        self._loss_guard_compress_ratio = float(os.getenv("EXIT_LOSS_GUARD_COMPRESS_RATIO", "0.7"))
        # MFE-based trail/partials for breakout・pullback系の尻尾切り
        self._mfe_partial_macro = float(os.getenv("EXIT_MFE_PARTIAL_MACRO", "8.0"))
        self._mfe_partial_micro = float(os.getenv("EXIT_MFE_PARTIAL_MICRO", "5.0"))
        self._mfe_trail_floor = float(os.getenv("EXIT_MFE_TRAIL_FLOOR", "3.5"))
        self._mfe_trail_gap = float(os.getenv("EXIT_MFE_TRAIL_GAP", "4.0"))

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
        range_mode: bool = False,
        now: Optional[datetime] = None,
        stage_tracker: Optional["StageTracker"] = None,
    ) -> List[ExitDecision]:
        current_time = self._ensure_utc(now)
        decisions: List[ExitDecision] = []
        try:
            close_price = float(fac_m1.get("close"))
        except (TypeError, ValueError):
            close_price = None
        if close_price is None:
            # Price不明なら安全側でスキップ
            return decisions
        projection_m1 = compute_ma_projection(fac_m1, timeframe_minutes=1.0)
        projection_h4 = compute_ma_projection(fac_h4, timeframe_minutes=240.0)
        atr_pips = fac_m1.get("atr_pips")
        if atr_pips is None:
            atr_pips = (fac_m1.get("atr") or 0.0) * 100.0
        close_price = float(fac_m1.get("close") or 0.0)
        for pocket, info in open_positions.items():
            if pocket == "__net__":
                continue
            long_units = int(info.get("long_units", 0) or 0)
            short_units = int(info.get("short_units", 0) or 0)
            avg_long = info.get("long_avg_price") or info.get("avg_price")
            avg_short = info.get("short_avg_price") or info.get("avg_price")
            long_profit = None
            short_profit = None
            if avg_long and close_price:
                long_profit = (close_price - avg_long) / 0.01
            if avg_short and close_price:
                short_profit = (avg_short - close_price) / 0.01
            if long_units == 0:
                self._reset_reverse_counter(pocket, "long")
            if short_units == 0:
                self._reset_reverse_counter(pocket, "short")
            if long_units == 0 and short_units == 0:
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
            ema20 = fac_m1.get("ema20", 0.0)
            projection_primary = projection_h4 if pocket == "macro" else projection_m1
            projection_fast = projection_m1

            if long_units > 0:
                if long_profit is not None and long_profit < 0:
                    continue  # 明示要求: マイナス時はEXITしない
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
                    ema20,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
                    stage_tracker,
                )
                if decision:
                    decisions.append(decision)

            if short_units > 0:
                if short_profit is not None and short_profit < 0:
                    continue  # 明示要求: マイナス時はEXITしない
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
                    ema20,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
                    stage_tracker,
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

    def _macro_loss_cap(self, atr_pips: float, matured: bool) -> float:
        atr_ref = float(atr_pips or 0.0)
        if atr_ref <= 0.0:
            atr_ref = 8.0
        value = atr_ref * (0.08 if matured else 0.06)
        return max(0.8, min(1.6, value))

    def _macro_slowdown_detected(
        self,
        *,
        profit_pips: float,
        adx: float,
        rsi: float,
        projection_fast: Optional[MACrossProjection],
        close_price: float,
        ema20: float,
        atr_pips: float,
    ) -> bool:
        if profit_pips < 1.2 or close_price is None or ema20 is None:
            return False
        buffer = max(0.8, (atr_pips or 6.0) * 0.35)
        momentum_band = (profit_pips <= max(8.0, (atr_pips or 6.0) * 1.2))
        ema_gap_pips = (close_price - ema20) / 0.01
        ema_cooling = ema_gap_pips <= buffer
        adx_fade = adx <= self._macro_trend_adx
        rsi_fade = rsi <= 58.0
        slope_fade = True
        if projection_fast is not None:
            slope_fade = projection_fast.gap_slope_pips < 0.08
        return momentum_band and ema_cooling and slope_fade and (adx_fade or rsi_fade)

    def _mfe_partial_units(self, pocket: str, units: int, profit_pips: float) -> int:
        """Return partial units when MFEが一定以上に達したときに利益を確定する。"""
        if units <= 0:
            return 0
        threshold = self._mfe_partial_macro if pocket == "macro" else self._mfe_partial_micro
        if profit_pips >= threshold:
            # take half but keep at least 1000 units
            partial = max(1000, units // 2)
            if partial < units:
                return partial
        return 0

    def _mfe_trail_hit(
        self,
        *,
        side: str,
        avg_price: Optional[float],
        close_price: Optional[float],
        profit_pips: float,
    ) -> bool:
        """Simple BE+ trail: once profit exceeds trail_gap, do not give back more than trail_floor."""
        if avg_price is None or close_price is None:
            return False
        # only arm when profit already healthy
        if profit_pips < self._mfe_trail_gap:
            return False
        cushion = max(1.0, self._mfe_trail_floor)
        give_back = profit_pips - cushion
        if give_back <= 0:
            return False
        if side == "long":
            trail_floor = avg_price + give_back * 0.01
            return close_price <= trail_floor
        trail_floor = avg_price - give_back * 0.01
        return close_price >= trail_floor

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
        ema20: float,
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        fac_m1: Dict,
        stage_tracker: Optional["StageTracker"],
    ) -> Optional[ExitDecision]:

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-long"
        avg_price = open_info.get("long_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (close_price - avg_price) / 0.01
        # ユーザー指定: マイナス時はEXITしない（SLなし運用のため、逆行中はホールド）
        if profit_pips < 0:
            return None
        neg_exit_blocked = self._negative_exit_blocked(
            pocket, open_info, "long", now, profit_pips, stage_tracker, atr_pips, fac_m1
        )
        target_bounds = self._entry_target_bounds(open_info, "long")
        if (
            reverse_signal
            and target_bounds
            and profit_pips is not None
            and profit_pips >= 0.0
            and profit_pips < target_bounds[0] * 0.75
        ):
            self._record_target_guard(
                pocket,
                "long",
                profit_pips,
                target_bounds,
                reverse_signal.get("tag") if reverse_signal else None,
            )
            reverse_signal = None

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

        matured_macro = False
        loss_cap = None
        if pocket == "macro":
            if (
                reverse_signal
                and profit_pips >= 4.0
                and close_price is not None
                and ema20 is not None
                and close_price >= ema20 + 0.002
            ):
                if (
                    adx >= self._macro_trend_adx + 4
                    or ma_gap_pips <= self._macro_ma_gap
                    or slope_support
                    or cross_support
                ):
                    reverse_signal = None
            matured_macro = self._has_mature_trade(
                open_info, "long", now, self._macro_min_hold_minutes
            )
            loss_cap = self._macro_loss_cap(atr_pips, matured_macro)

        ema_gap_pips = None
        if close_price is not None and ema20 is not None:
            ema_gap_pips = (close_price - ema20) / 0.01

        if pocket == "macro":
            partial_units = self._trendma_partial_exit_units(
                open_info=open_info,
                side="long",
                units=units,
                profit_pips=profit_pips,
                adx=adx,
                rsi=rsi,
                projection_fast=projection_fast,
                atr_pips=atr_pips,
                loss_cap=loss_cap,
            )
            if partial_units:
                return ExitDecision(
                    pocket=pocket,
                    units=partial_units,
                    reason="trendma_partial",
                    tag="trendma-decay",
                    allow_reentry=True,
                )
            if (
                close_price is not None
                and ema20 is not None
                and self._macro_slowdown_detected(
                    profit_pips=profit_pips,
                    adx=adx,
                    rsi=rsi,
                    projection_fast=projection_fast,
                    close_price=close_price,
                    ema20=ema20,
                    atr_pips=atr_pips,
                )
            ):
                partial_units = max(1000, (units // 2))
                if partial_units < units:
                    return ExitDecision(
                        pocket=pocket,
                        units=-partial_units,
                        reason="macro_slowdown",
                        tag="macro-slowdown",
                        allow_reentry=True,
                    )
            vol_partial = self._vol_partial_exit_units(
                pocket=pocket,
                side="long",
                units=units,
                profit_pips=profit_pips,
                atr_pips=atr_pips,
                ema_gap_pips=ema_gap_pips,
            )
            if vol_partial:
                return ExitDecision(
                    pocket=pocket,
                    units=vol_partial,
                    reason="macro_vol_partial",
                    tag="macro-vol-partial",
                    allow_reentry=True,
                )

        # Generic MFE-based partial/trail for breakout/pullback styles
        mfe_partial = self._mfe_partial_units(pocket, units, profit_pips)
        if mfe_partial:
            return ExitDecision(
                pocket=pocket,
                units=mfe_partial,
                reason="mfe_partial",
                tag=f"{pocket}-mfe-partial",
                allow_reentry=True,
            )
        if self._mfe_trail_hit(
            side="long",
            avg_price=avg_price,
            close_price=close_price,
            profit_pips=profit_pips,
        ):
            return ExitDecision(
                pocket=pocket,
                units=units,
                reason="mfe_trail",
                tag=f"{pocket}-mfe-trail",
                allow_reentry=False,
            )

        # Do not act on fresh macro trades unless conditions are clearly adverse
        if reverse_signal and pocket == "macro" and not range_mode:
            if not matured_macro:
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

        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                # Pattern-aware retest guard for macro: if price sits near fast MA
                # on lower TFs and M5/M10 slopes support a bounce, defer exit.
                if pocket == "macro":
                    if self._should_delay_macro_exit_for_retest(
                        side="long",
                        close_price=close_price,
                        ema20=ema20,
                        atr_pips=atr_pips,
                        fac_m1=fac_m1,
                    ):
                        return None
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.2 < profit_pips < 1.2)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
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
        elif pocket == "scalp" and close_price > ema20:
            reason = "scalp_momentum_flip"
        elif (
            pocket == "micro"
            and profit_pips <= -self._micro_loss_grace_pips
            and self._micro_loss_ready(open_info, "long", now)
        ):
            reason = "micro_loss_guard"
        elif pocket == "macro" and loss_cap is not None and profit_pips <= -loss_cap:
            reason = "macro_loss_cap"
        elif (
            pocket == "macro"
            and avg_price
            and atr_pips is not None
            and profit_pips >= max(3.5, atr_pips * 0.9)
        ):
            trail_back = max(1.6, atr_pips * 0.45)
            trail_floor = avg_price + (profit_pips - trail_back) * 0.01
            if close_price is not None and close_price <= trail_floor:
                reason = "macro_atr_trail"
        elif self._should_exit_for_cross(
            pocket,
            "long",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
            atr_pips,
        ):
            reason = "ma_cross_imminent"
        elif (
            pocket == "macro"
            and profit_pips >= max(4.2, atr_pips * 1.0)
            and close_price is not None
            and ema20 is not None
            and close_price <= ema20 - max(0.0010, (atr_pips * 0.25) / 100)
        ):
            reason = "macro_trend_fade"
        elif pocket == "micro" and self._micro_profit_exit_ready(
            side="long",
            profit_pips=profit_pips,
            rsi=rsi,
            close_price=close_price,
            ema20=ema20,
            projection_fast=projection_fast,
        ):
            reason = "micro_profit_guard"
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if (
                pocket == "macro"
                and self._has_mature_trade(open_info, "long", now, self._range_macro_grace_minutes)
            ):
                return None
            if profit_pips >= 1.6:
                reason = "range_take_profit"
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"
        elif self._ema_release_ready(
            pocket=pocket,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            close_price=close_price,
            ema20=ema20,
        ):
            reason = "macro_ema_release"
        elif self._profit_snatch_ready(
            pocket=pocket,
            side="long",
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        ):
            reason = "micro_profit_snatch"
            allow_reentry = True

        # Structure-based kill-line: if macro and M5 pivot breaks beyond cushion, exit decisively
        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="long", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

        if reason and pocket in {"micro", "scalp"} and reason != "event_lock":
            if neg_exit_blocked:
                return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        return ExitDecision(
            pocket=pocket,
            units=-abs(units),
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
        ema20: float,
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        fac_m1: Dict,
        stage_tracker: Optional["StageTracker"],
    ) -> Optional[ExitDecision]:

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-short"
        avg_price = open_info.get("short_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (avg_price - close_price) / 0.01
        # ユーザー指定: マイナス時はEXITしない（SLなし運用のため、逆行中はホールド）
        if profit_pips < 0:
            return None
        neg_exit_blocked = self._negative_exit_blocked(
            pocket, open_info, "short", now, profit_pips, stage_tracker, atr_pips, fac_m1
        )
        target_bounds = self._entry_target_bounds(open_info, "short")
        if (
            reverse_signal
            and target_bounds
            and profit_pips is not None
            and profit_pips >= 0.0
            and profit_pips < target_bounds[0] * 0.75
        ):
            self._record_target_guard(
                pocket,
                "short",
                profit_pips,
                target_bounds,
                reverse_signal.get("tag") if reverse_signal else None,
            )
            reverse_signal = None

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

        matured_macro = False
        loss_cap = None
        if pocket == "macro":
            if (
                reverse_signal
                and profit_pips >= 4.0
                and close_price is not None
                and ema20 is not None
                and close_price <= ema20 - 0.002
            ):
                if (
                    adx >= self._macro_trend_adx + 4
                    or ma_gap_pips <= self._macro_ma_gap
                    or slope_support
                    or cross_support
                ):
                    reverse_signal = None
            matured_macro = self._has_mature_trade(
                open_info, "short", now, self._macro_min_hold_minutes
            )
            loss_cap = self._macro_loss_cap(atr_pips, matured_macro)

        ema_gap_pips = None
        if close_price is not None and ema20 is not None:
            ema_gap_pips = (close_price - ema20) / 0.01

        if pocket == "macro":
            partial_units = self._trendma_partial_exit_units(
                open_info=open_info,
                side="short",
                units=units,
                profit_pips=profit_pips,
                adx=adx,
                rsi=rsi,
                projection_fast=projection_fast,
                atr_pips=atr_pips,
                loss_cap=loss_cap,
            )
            if partial_units:
                return ExitDecision(
                    pocket=pocket,
                    units=partial_units,
                    reason="trendma_partial",
                    tag="trendma-decay",
                    allow_reentry=True,
                )
            vol_partial = self._vol_partial_exit_units(
                pocket=pocket,
                side="short",
                units=units,
                profit_pips=profit_pips,
                atr_pips=atr_pips,
                ema_gap_pips=ema_gap_pips,
            )
            if vol_partial:
                return ExitDecision(
                    pocket=pocket,
                    units=vol_partial,
                    reason="macro_vol_partial",
                    tag="macro-vol-partial",
                    allow_reentry=True,
                )

        mfe_partial = self._mfe_partial_units(pocket, units, profit_pips)
        if mfe_partial:
            return ExitDecision(
                pocket=pocket,
                units=mfe_partial,
                reason="mfe_partial",
                tag=f"{pocket}-mfe-partial",
                allow_reentry=True,
            )
        if self._mfe_trail_hit(
            side="short",
            avg_price=avg_price,
            close_price=close_price,
            profit_pips=profit_pips,
        ):
            return ExitDecision(
                pocket=pocket,
                units=units,
                reason="mfe_trail",
                tag=f"{pocket}-mfe-trail",
                allow_reentry=False,
            )

        if (
            pocket == "micro"
            and profit_pips <= -4.0
            and close_price is not None
            and ema20 is not None
            and (
                close_price >= ema20 + 0.0015
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
                if pocket == "macro":
                    if self._should_delay_macro_exit_for_retest(
                        side="short",
                        close_price=close_price,
                        ema20=ema20,
                        atr_pips=atr_pips,
                        fac_m1=fac_m1,
                    ):
                        return None
                # ヒステリシス: 小幅の含み損益域では逆方向シグナルだけでクローズしない
                micro_guard = (pocket == "micro" and -1.2 < profit_pips < 1.2)
                macro_guard = (
                    pocket == "macro"
                    and -self._macro_hysteresis_pips < profit_pips < self._macro_hysteresis_pips
                )
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
        elif (
            pocket == "micro"
            and profit_pips <= -self._micro_loss_grace_pips
            and self._micro_loss_ready(open_info, "short", now)
        ):
            reason = "micro_loss_guard"
        elif pocket == "macro" and loss_cap is not None and profit_pips <= -loss_cap:
            reason = "macro_loss_cap"
        elif pocket == "scalp" and close_price < ema20:
            reason = "scalp_momentum_flip"
        elif (
            pocket == "macro"
            and avg_price
            and profit_pips >= max(6.0, atr_pips * 1.05)
        ):
            trail_back = max(2.8, atr_pips * 0.55)
            trail_ceiling = avg_price - (profit_pips - trail_back) * 0.01
            if close_price is not None and close_price >= trail_ceiling:
                reason = "macro_atr_trail"
        elif self._should_exit_for_cross(
            pocket,
            "short",
            open_info,
            projection_primary,
            projection_fast,
            profit_pips,
            now,
            macd_cross_minutes,
            atr_pips,
        ):
            reason = "ma_cross_imminent"
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif (
            pocket == "macro"
            and profit_pips >= max(6.5, atr_pips * 1.1)
            and close_price is not None
            and ema20 is not None
            and close_price >= ema20 + max(0.0012, (atr_pips * 0.3) / 100)
        ):
            reason = "macro_trend_fade"
        elif pocket == "micro" and self._micro_profit_exit_ready(
            side="short",
            profit_pips=profit_pips,
            rsi=rsi,
            close_price=close_price,
            ema20=ema20,
            projection_fast=projection_fast,
        ):
            reason = "micro_profit_guard"
        elif range_mode:
            if (
                pocket == "macro"
                and self._has_mature_trade(open_info, "short", now, self._range_macro_grace_minutes)
            ):
                return None
            if profit_pips >= 1.6:
                reason = "range_take_profit"
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"
        elif self._ema_release_ready(
            pocket=pocket,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            close_price=close_price,
            ema20=ema20,
        ):
            reason = "macro_ema_release"
        elif self._profit_snatch_ready(
            pocket=pocket,
            side="short",
            open_info=open_info,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            fac_m1=fac_m1,
            now=now,
        ):
            reason = "micro_profit_snatch"
            allow_reentry = True

        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="short", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

        if reason and pocket in {"micro", "scalp"} and reason != "event_lock":
            if neg_exit_blocked:
                return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason == "reverse_signal":
            allow_reentry = False
        if not reason:
            return None

        return ExitDecision(
            pocket=pocket,
            units=abs(units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

    # --- Pattern-aware helpers (macro) ---
    def _should_delay_macro_exit_for_retest(
        self,
        *,
        side: str,
        close_price: float,
        ema20: float,
        atr_pips: float,
        fac_m1: Dict,
    ) -> bool:
        """Return True to defer macro exit for a potential retest/bounce.

        Logic:
        - If price sits near fast MA band (M5/M10 via projection fast MA),
          and the corresponding slopes align with the position, hold.
        - Band width scales with ATR.
        """
        try:
            candles_m1 = fac_m1.get("candles") or []
            c5 = resample_candles_from_m1(candles_m1, 5)
            c10 = resample_candles_from_m1(candles_m1, 10)
            p5 = compute_ma_projection({"candles": c5}, timeframe_minutes=5.0) if len(c5) >= 30 else None
            p10 = compute_ma_projection({"candles": c10}, timeframe_minutes=10.0) if len(c10) >= 30 else None
        except Exception:
            return False

        dir_sign = 1.0 if side == "long" else -1.0
        slope_ok_5 = p5 and (p5.gap_slope_pips or 0.0) * dir_sign >= self._macro_retest_m5_slope
        slope_ok_10 = p10 and (p10.gap_slope_pips or 0.0) * dir_sign >= self._macro_retest_m10_slope
        if not (slope_ok_5 or slope_ok_10):
            return False

        band = max(self._macro_retest_band_base, (atr_pips or 0.0) * 0.25)
        # Prefer M5 fast MA proximity when available; fall back to ema20 (M1)
        near_fast_ok = False
        try:
            fast_approx = (p5.fast_ma if p5 and p5.fast_ma is not None else None)
            ref = fast_approx if fast_approx is not None else ema20
            near_fast_ok = abs(close_price - ref) / 0.01 <= band
        except Exception:
            near_fast_ok = abs(close_price - ema20) / 0.01 <= band
        return bool(near_fast_ok)

    def _structure_break_if_any(
        self,
        *,
        side: str,
        fac_m1: Dict,
        price: float,
        atr_pips: float,
    ) -> Optional[str]:
        """Detect recent M5 pivot break as decisive structure failure.

        Returns reason string when broken; otherwise None.
        """
        try:
            candles_m1 = fac_m1.get("candles") or []
            c5 = resample_candles_from_m1(candles_m1, 5)
        except Exception:
            return None
        if len(c5) < 9:
            return None

        low_key, high_key = ("low", "high")
        def _last_pivot(arr: List[Dict[str, float]], is_low: bool, width: int = 2, lookback: int = 14) -> Optional[float]:
            n = len(arr)
            start = max(2, n - lookback)
            for i in range(n - 3, start - 1, -1):
                try:
                    center = float(arr[i][low_key if is_low else high_key])
                except Exception:
                    continue
                ok = True
                for k in range(1, width + 1):
                    try:
                        left = float(arr[i - k][low_key if is_low else high_key])
                        right = float(arr[i + k][low_key if is_low else high_key])
                    except Exception:
                        ok = False
                        break
                    if is_low:
                        if not (center <= left and center <= right):
                            ok = False
                            break
                    else:
                        if not (center >= left and center >= right):
                            ok = False
                            break
                if ok:
                    return center
            return None

        cushion = max(0.6, (atr_pips or 0.0) * self._macro_struct_cushion)
        if side == "long":
            pivot = _last_pivot(c5, is_low=True)
            if pivot is not None and price <= pivot - cushion * 0.01:
                return "macro_struct_break"
        else:
            pivot = _last_pivot(c5, is_low=False)
            if pivot is not None and price >= pivot + cushion * 0.01:
                return "macro_struct_break"
        return None

    def _micro_loss_ready(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
    ) -> bool:
        if self._micro_loss_hold_seconds <= 0:
            return True
        youngest = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        return youngest >= self._micro_loss_hold_seconds

    def _micro_profit_exit_ready(
        self,
        *,
        side: str,
        profit_pips: float,
        rsi: float,
        close_price: float,
        ema20: float,
        projection_fast: Optional[MACrossProjection],
    ) -> bool:
        if profit_pips is None:
            return False
        hard = profit_pips >= self._micro_profit_hard
        soft = profit_pips >= self._micro_profit_soft
        if not (hard or soft):
            return False

        ema_trigger = False
        if close_price is not None and ema20 is not None and self._micro_profit_ema_buffer > 0:
            gap = close_price - ema20
            if side == "long" and gap <= -self._micro_profit_ema_buffer:
                ema_trigger = True
            elif side == "short" and gap >= self._micro_profit_ema_buffer:
                ema_trigger = True

        rsi_trigger = False
        try:
            rsi_val = float(rsi)
        except (TypeError, ValueError):
            rsi_val = None
        if rsi_val is not None:
            if side == "long" and rsi_val <= self._micro_profit_rsi_release_long:
                rsi_trigger = True
            elif side == "short" and rsi_val >= self._micro_profit_rsi_release_short:
                rsi_trigger = True

        slope_trigger = False
        slope = None
        if projection_fast is not None:
            try:
                slope = float(projection_fast.gap_slope_pips or 0.0)
            except Exception:
                slope = None
        if slope is not None and self._micro_profit_slope_min > 0:
            if side == "long" and slope <= -self._micro_profit_slope_min:
                slope_trigger = True
            elif side == "short" and slope >= self._micro_profit_slope_min:
                slope_trigger = True

        if hard and (ema_trigger or rsi_trigger or slope_trigger):
            return True
        if soft and ema_trigger and (rsi_trigger or slope_trigger):
            return True
        return False

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
        atr_pips: float,
    ) -> bool:
        # For macro positions, prefer primary (H4) projection to avoid
        # churning exits on transient M1 flickers. For other pockets,
        # keep fast (M1) responsive.
        projection = (
            projection_primary if pocket == "macro" else (projection_fast or projection_primary)
        )
        if projection is None:
            return False

        gap = projection.gap_pips
        if side == "long" and gap < 0.0:
            return True
        if side == "short" and gap > 0.0:
            return True

        slope_source = (
            projection_primary if pocket == "macro" else (projection_fast or projection_primary)
        )
        slope = slope_source.gap_slope_pips if slope_source else 0.0
        if macd_cross_minutes is None:
            if side == "long" and slope >= 0.0:
                return False
            if side == "short" and slope <= 0.0:
                return False

        threshold = 3.5
        matured = False
        loss_guard = 0.0
        small_profit_guard = 0.0
        if pocket == "macro":
            matured = self._has_mature_trade(
                open_info, side, now, self._macro_min_hold_minutes
            )
            threshold = 7.0 if matured else 4.8
            atr_ref = float(atr_pips or 0.0)
            if atr_ref <= 0.0:
                atr_ref = 8.0
            loss_guard = atr_ref * (0.16 if matured else 0.12)
            loss_guard = max(0.9, min(1.8, loss_guard))
            small_profit_guard = max(0.5, min(1.4, loss_guard * 0.75))
        elif pocket == "scalp":
            threshold = 2.2

        candidates: List[float] = []
        if projection.projected_cross_minutes is not None:
            candidates.append(projection.projected_cross_minutes)
        if macd_cross_minutes is not None:
            candidates.append(macd_cross_minutes)
        # When no reliable cross projection is available, fall back to
        # simple loss/take-profit guards. For macro positions, only stop
        # on loss beyond the guard, or realize a small profit when it
        # exceeds the small_profit_guard. Do NOT exit simply because the
        # profit is small or slightly negative.
        if not candidates:
            if pocket == "macro":
                if profit_pips <= -loss_guard:
                    return True  # stop loss guard
                if profit_pips >= small_profit_guard:
                    return True  # small take-profit
            return False
        cross_minutes = min(candidates)

        if cross_minutes > threshold:
            if pocket == "macro" and (
                profit_pips <= -loss_guard or profit_pips >= small_profit_guard
            ):
                return True
            return False

        if pocket == "macro":
            if not matured:
                return profit_pips <= -(loss_guard * 1.1)
            if profit_pips <= -loss_guard:
                return True  # stop loss
            if profit_pips >= small_profit_guard:
                return True  # small take-profit once threshold reached
            return False

        if profit_pips >= 0.8:
            return True
        if pocket == "macro" and not matured and cross_minutes <= threshold / 2.0:
            return False
        if cross_minutes <= threshold / 2.0:
            return True
        if (
            macd_cross_minutes is not None
            and macd_cross_minutes <= threshold / 2.0
            and not (pocket == "macro" and not matured)
        ):
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

    def _youngest_trade_age_seconds(
        self,
        open_info: Dict,
        side: str,
        now: datetime,
    ) -> Optional[float]:
        trades = open_info.get("open_trades") or []
        youngest: Optional[float] = None
        for tr in trades:
            if tr.get("side") != side:
                continue
            opened_at = self._parse_open_time(tr.get("open_time"))
            if opened_at is None:
                continue
            age = (now - opened_at).total_seconds()
            if age < 0:
                continue
            if youngest is None or age < youngest:
                youngest = age
        return youngest

    def _trade_age_seconds(self, trade: Dict, now: datetime) -> Optional[float]:
        opened_at = self._parse_open_time(trade.get("open_time"))
        if opened_at is None:
            return None
        age = (now - opened_at).total_seconds()
        if age < 0:
            return 0.0
        return age

    def _negative_exit_blocked(
        self,
        pocket: str,
        open_info: Dict,
        side: str,
        now: datetime,
        profit_pips: float,
        stage_tracker: Optional["StageTracker"],
        atr_pips: float,
        fac_m1: Dict,
    ) -> bool:
        if profit_pips is None or profit_pips >= 0.0:
            return False
        default_loss, default_hold = self._default_loss_hold(pocket)
        trades = [
            tr
            for tr in (open_info.get("open_trades") or [])
            if tr.get("side") == side
        ]
        if not trades:
            return False
        blocked = False
        for tr in trades:
            thesis = self._parse_entry_thesis(tr)
            strategy_tag = thesis.get("strategy_tag") or tr.get("strategy_tag")
            loss_guard, hold_req = self._trade_guard_requirements(
                thesis, pocket, default_loss, default_hold
            )
            loss_guard = self._volatility_loss_clamp(
                pocket=pocket,
                loss_guard=loss_guard,
                atr_pips=atr_pips,
                fac_m1=fac_m1,
            )
            if loss_guard <= 0.0 or hold_req <= 0.0:
                continue
            if profit_pips <= -loss_guard:
                continue
            age = self._trade_age_seconds(tr, now)
            if age is None:
                continue
            if age < hold_req:
                trade_id = tr.get("trade_id")
                logging.info(
                    "[EXIT] hold_guard block pocket=%s side=%s trade=%s age=%.1fs req=%.1fs loss_guard=%.2fp profit=%.2fp",
                    pocket,
                    side,
                    trade_id,
                    age,
                    hold_req,
                    loss_guard,
                    profit_pips,
                )
                self._record_hold_violation(
                    pocket,
                    side,
                    strategy_tag,
                    hold_req,
                    age,
                    now,
                    stage_tracker,
                )
                blocked = True
                break
        return blocked

    def _default_loss_hold(self, pocket: str) -> tuple[float, float]:
        guard_map = {
            "micro": (self._micro_loss_grace_pips, self._micro_min_hold_seconds),
            "scalp": (self._scalp_loss_grace_pips, self._scalp_min_hold_seconds),
        }
        return guard_map.get(pocket, (0.0, 0.0))

    def _trade_guard_requirements(
        self,
        thesis: Dict,
        pocket: str,
        default_loss: float,
        default_hold: float,
    ) -> tuple[float, float]:
        loss_guard = thesis.get("loss_guard_pips") or thesis.get("loss_grace_pips")
        hold_req = thesis.get("min_hold_sec") or thesis.get("min_hold_seconds")
        try:
            loss_val = float(loss_guard)
        except (TypeError, ValueError):
            loss_val = default_loss
        if loss_val <= 0.0:
            loss_val = default_loss
        try:
            hold_val = float(hold_req)
        except (TypeError, ValueError):
            hold_val = default_hold
        if hold_val <= 0.0:
            hold_val = default_hold
        return max(0.0, loss_val), max(0.0, hold_val)

    def _volatility_loss_clamp(
        self,
        *,
        pocket: str,
        loss_guard: float,
        atr_pips: float,
        fac_m1: Dict,
    ) -> float:
        if loss_guard <= 0.0 or pocket not in {"micro", "scalp"}:
            return loss_guard
        atr_val = float(atr_pips or 0.0)
        try:
            vol_val = float(fac_m1.get("vol_5m") or 0.0)
        except (TypeError, ValueError):
            vol_val = 0.0
        if atr_val < self._loss_guard_atr_trigger and (vol_val <= 0.0 or vol_val < self._loss_guard_vol_trigger):
            return loss_guard
        clamped = max(0.2, loss_guard * self._loss_guard_compress_ratio)
        if clamped < loss_guard:
            logging.info(
                "[EXIT] loss_guard clamp pocket=%s atr=%.2f vol=%.2f %.2f->%.2f",
                pocket,
                atr_val,
                vol_val,
                loss_guard,
                clamped,
            )
        return clamped

    def _record_hold_violation(
        self,
        pocket: str,
        direction: str,
        strategy_tag: Optional[str],
        required_sec: float,
        actual_sec: float,
        now: datetime,
        stage_tracker: Optional["StageTracker"],
    ) -> None:
        tags = {
            "pocket": pocket,
            "direction": direction,
        }
        if strategy_tag:
            tags["strategy"] = strategy_tag
        log_metric("exit_hold_violation", 1.0, tags=tags)
        if stage_tracker:
            stage_tracker.log_hold_violation(
                pocket,
                direction,
                required_sec=required_sec,
                actual_sec=actual_sec,
                reason="hold_guard_block",
                cooldown_seconds=self._hold_violation_cooldown(pocket),
                now=now,
            )

    @staticmethod
    def _hold_violation_cooldown(pocket: str) -> int:
        if pocket == "macro":
            return 420
        if pocket == "micro":
            return 240
        return 180

    def _entry_target_bounds(
        self,
        open_info: Dict,
        side: str,
    ) -> Optional[tuple[float, float]]:
        trades = open_info.get("open_trades") or []
        targets: list[float] = []
        for tr in trades:
            if tr.get("side") != side:
                continue
            thesis = self._parse_entry_thesis(tr)
            target = thesis.get("target_tp_pips") or thesis.get("tp_hint_pips")
            try:
                if target is not None:
                    targets.append(float(target))
            except (TypeError, ValueError):
                continue
        if not targets:
            return None
        targets.sort()
        return targets[0], sum(targets) / len(targets)

    def _record_target_guard(
        self,
        pocket: str,
        direction: str,
        profit_pips: float,
        target_bounds: tuple[float, float],
        signal_tag: Optional[str],
    ) -> None:
        log_metric(
            "exit_target_guard",
            1.0,
            tags={
                "pocket": pocket,
                "direction": direction,
                "signal": signal_tag or "reverse",
            },
        )
        logging.info(
            "[EXIT] target_guard pocket=%s dir=%s profit=%.2fp guard<=%.2fp",
            pocket,
            direction,
            profit_pips,
            target_bounds[0],
        )

    @staticmethod
    def _parse_entry_thesis(trade: Dict) -> Dict:
        thesis = trade.get("entry_thesis") or {}
        if isinstance(thesis, str):
            try:
                thesis = json.loads(thesis)
            except Exception:
                thesis = {}
        if not isinstance(thesis, dict):
            thesis = {}
        return thesis

    def _has_strategy(
        self,
        open_info: Dict,
        strategy_keyword: str,
        side: Optional[str] = None,
    ) -> bool:
        trades = open_info.get("open_trades") or []
        for tr in trades:
            if side and tr.get("side") != side:
                continue
            tag = tr.get("strategy_tag")
            if isinstance(tag, str) and strategy_keyword in tag:
                return True
            thesis = self._parse_entry_thesis(tr)
            strat = thesis.get("strategy_tag") or thesis.get("strategy")
            if isinstance(strat, str) and strategy_keyword in strat:
                return True
        return False

    def _trendma_partial_exit_units(
        self,
        *,
        open_info: Dict,
        side: str,
        units: int,
        profit_pips: float,
        adx: float,
        rsi: float,
        projection_fast: Optional[MACrossProjection],
        atr_pips: float,
        loss_cap: Optional[float],
    ) -> Optional[int]:
        if not self._has_strategy(open_info, "TrendMA", side):
            return None
        if profit_pips is None or profit_pips <= 0.25:
            return None
        atr_val = float(atr_pips or 0.0)
        cap = loss_cap or max(1.8, atr_val * 0.8)
        profit_ceiling = max(self._trendma_partial_profit_cap, cap * 1.25)
        if profit_pips > profit_ceiling:
            return None
        slope = None
        if projection_fast is not None:
            try:
                slope = float(projection_fast.gap_slope_pips or 0.0)
            except Exception:
                slope = None
        slope_fade = False
        if slope is not None:
            if side == "long":
                slope_fade = slope <= 0.02
            else:
                slope_fade = slope >= -0.02
        adx_fade = adx <= self._macro_trend_adx + 1.5
        if side == "long":
            rsi_fade = rsi <= 56.0
        else:
            rsi_fade = rsi >= 44.0
        if not (slope_fade or (adx_fade and rsi_fade)):
            return None
        reduce_units = max(1000, int(abs(units) * self._trendma_partial_fraction))
        if reduce_units >= abs(units):
            reduce_units = abs(units) - 1000
        if reduce_units <= 0:
            return None
        return -reduce_units if side == "long" else reduce_units

    def _vol_partial_exit_units(
        self,
        *,
        pocket: str,
        side: str,
        units: int,
        profit_pips: float,
        atr_pips: float,
        ema_gap_pips: Optional[float],
    ) -> Optional[int]:
        if pocket != "macro":
            return None
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._vol_partial_atr_min:
            return None
        profit_floor = max(self._vol_partial_profit_floor, atr_val * 1.5)
        if self._vol_partial_profit_cap > 0:
            profit_floor = min(profit_floor, self._vol_partial_profit_cap)
        if profit_pips is None or profit_pips < profit_floor:
            return None
        if ema_gap_pips is not None and abs(ema_gap_pips) > atr_val * 2.5:
            return None
        fraction = self._vol_partial_fraction
        if atr_val >= self._vol_partial_atr_max:
            fraction = max(0.5, fraction * 0.85)
        reduce_units = max(1000, int(abs(units) * fraction))
        if reduce_units >= abs(units):
            reduce_units = abs(units) - 1000
        if reduce_units <= 0:
            return None
        return -reduce_units if side == "long" else reduce_units

    def _ema_release_ready(
        self,
        *,
        pocket: str,
        profit_pips: float,
        atr_pips: float,
        close_price: float,
        ema20: float,
    ) -> bool:
        if pocket != "macro":
            return False
        if profit_pips is None or profit_pips <= 0.0:
            return False
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._vol_partial_atr_min:
            return False
        if close_price is None or ema20 is None:
            return False
        ema_gap = (close_price - ema20) / 0.01
        return abs(ema_gap) <= self._vol_ema_release_gap

    def _profit_snatch_ready(
        self,
        *,
        pocket: str,
        side: str,
        open_info: Dict,
        profit_pips: float,
        atr_pips: float,
        fac_m1: Dict,
        now: datetime,
    ) -> bool:
        if pocket not in {"macro", "micro"}:
            return False
        if profit_pips is None:
            return False
        if not (self._profit_snatch_min <= profit_pips <= self._profit_snatch_max):
            return False
        atr_val = float(atr_pips or 0.0)
        if atr_val < self._profit_snatch_atr_min:
            return False
        vol = fac_m1.get("vol_5m")
        try:
            vol_val = float(vol) if vol is not None else None
        except (TypeError, ValueError):
            vol_val = None
        if vol_val is not None and vol_val < self._profit_snatch_vol_min:
            return False
        if not _in_jst_window(now, self._profit_snatch_jst_start, self._profit_snatch_jst_end):
            return False
        age = self._youngest_trade_age_seconds(open_info, side, now) or 0.0
        return age >= self._profit_snatch_hold

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
