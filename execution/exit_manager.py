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
        self._macro_loss_buffer = 4.0
        self._macro_ma_gap = 3.0
        # Macro-specific stability controls
        self._macro_min_hold_minutes = 12.0  # more patience before acting on reversals
        self._macro_hysteresis_pips = 5.0    # wider no-close band to avoid jitter
        # Micro-specific minimum hold
        self._micro_min_hold_minutes = 6.0
        # MFE guard: if we have achieved decent MFE, allow deeper pullbacks before exiting
        self._mfe_guard_pips = 1.6
        self._mfe_guard_retrace_ratio = 0.62
        self._reverse_confirmations = 3
        self._reverse_decay = timedelta(seconds=300)
        self._reverse_hits: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._range_macro_grace_minutes = 10.0

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
        range_mode: bool = False,
        now: Optional[datetime] = None,
        story: Optional[ChartStorySnapshot] = None,
    ) -> List[ExitDecision]:
        current_time = self._ensure_utc(now)
        decisions: List[ExitDecision] = []
        projection_m1 = compute_ma_projection(fac_m1, timeframe_minutes=1.0)
        projection_h4 = compute_ma_projection(fac_h4, timeframe_minutes=240.0)
        m1_candles = fac_m1.get("candles") or []
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
                    m1_candles,
                    story,
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
                    m1_candles,
                    story,
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
        ema20: float,
        range_mode: bool,
        now: datetime,
        projection_primary: Optional[MACrossProjection],
        projection_fast: Optional[MACrossProjection],
        m1_candles: List[Dict],
        story: Optional[ChartStorySnapshot],
    ) -> Optional[ExitDecision]:
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

        if (
            reverse_signal
            and pocket == "macro"
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
        elif pocket == "scalp" and close_price > ema20:
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
            and profit_pips >= 6.0
            and close_price is not None
            and ema20 is not None
            and close_price <= ema20 - 0.0015
        ):
            reason = "macro_trail_hit"
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

        # MFE-based patience: if we've achieved decent favorable excursion,
        # avoid exiting on a mild pullback unless strong invalidation.
        if reason == "reverse_signal" and not range_mode:
            max_mfe = self._max_mfe_for_side(open_info, "long", m1_candles, now)
            if max_mfe is not None and max_mfe >= self._mfe_guard_pips:
                retrace = max(0.0, max_mfe - max(0.0, profit_pips))
                if retrace <= self._mfe_guard_retrace_ratio * max_mfe and profit_pips > -self._macro_loss_buffer:
                    return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason:
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

        self._record_exit_metric(
            pocket,
            "long",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

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
        m1_candles: List[Dict],
        story: Optional[ChartStorySnapshot],
    ) -> Optional[ExitDecision]:
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

        if (
            reverse_signal
            and pocket == "macro"
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
        elif pocket == "scalp" and close_price < ema20:
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

        if reason == "reverse_signal" and not range_mode:
            max_mfe = self._max_mfe_for_side(open_info, "short", m1_candles, now)
            if max_mfe is not None and max_mfe >= self._mfe_guard_pips:
                retrace = max(0.0, max_mfe - max(0.0, profit_pips))
                if retrace <= self._mfe_guard_retrace_ratio * max_mfe and profit_pips > -self._macro_loss_buffer:
                    return None

        if range_mode and reason == "reverse_signal":
            allow_reentry = False
        if reason:
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

        self._record_exit_metric(
            pocket,
            "short",
            reason,
            profit_pips,
            story,
            range_mode,
            now,
        )

        return ExitDecision(
            pocket=pocket,
            units=abs(units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )

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
        if pocket == "macro":
            threshold = 10.0
            if not self._has_mature_trade(open_info, side, now, self._macro_min_hold_minutes):
                threshold = 5.0
        elif pocket == "scalp":
            threshold = 2.2

        if cross_minutes > threshold:
            return False

        if pocket == "macro" and profit_pips <= -self._macro_loss_buffer:
            return True
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
