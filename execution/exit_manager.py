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
from typing import Dict, List, Optional, Tuple

import logging

from analysis.ma_projection import MACrossProjection, compute_ma_projection
from analysis.mtf_utils import resample_candles_from_m1


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

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
        range_mode: bool = False,
        now: Optional[datetime] = None,
    ) -> List[ExitDecision]:
        current_time = self._ensure_utc(now)
        decisions: List[ExitDecision] = []
        projection_m1 = compute_ma_projection(fac_m1, timeframe_minutes=1.0)
        projection_h4 = compute_ma_projection(fac_h4, timeframe_minutes=240.0)
        atr_pips = fac_m1.get("atr_pips")
        if atr_pips is None:
            atr_pips = (fac_m1.get("atr") or 0.0) * 100.0
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
            close_price = fac_m1.get("close", 0.0)
            projection_primary = projection_h4 if pocket == "macro" else projection_m1
            projection_fast = projection_m1

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
                    ema20,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
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
                    ema20,
                    range_mode,
                    current_time,
                    projection_primary,
                    projection_fast,
                    atr_pips,
                    fac_m1,
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
    ) -> Optional[ExitDecision]:
        if pocket == "scalp":
            return None

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-long"
        avg_price = open_info.get("long_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (close_price - avg_price) / 0.01

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

        # Structure-based kill-line: if macro and M5 pivot breaks beyond cushion, exit decisively
        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="long", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

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
    ) -> Optional[ExitDecision]:
        if pocket == "scalp":
            return None

        allow_reentry = False
        reason = ""
        tag = f"{pocket}-short"
        avg_price = open_info.get("short_avg_price") or open_info.get("avg_price")
        profit_pips = 0.0
        if avg_price and close_price:
            profit_pips = (avg_price - close_price) / 0.01

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

        if pocket == "macro" and not reason:
            kill_reason = self._structure_break_if_any(
                side="short", fac_m1=fac_m1, price=close_price, atr_pips=atr_pips
            )
            if kill_reason:
                reason = kill_reason

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
