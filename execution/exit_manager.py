"""
execution.exit_manager
~~~~~~~~~~~~~~~~~~~~~~
注文のクローズ判定を担当。
• 逆方向シグナル or 指標の劣化を検知してクローズ指示を返す
• イベント時のポケット縮退もここでハンドル
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


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

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
        range_mode: bool = False,
    ) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        for pocket, info in open_positions.items():
            if pocket == "__net__":
                continue
            long_units = int(info.get("long_units", 0) or 0)
            short_units = int(info.get("short_units", 0) or 0)
            if long_units == 0 and short_units == 0:
                continue

            reverse_short = self._strong_signal(signals, pocket, "OPEN_SHORT")
            reverse_long = self._strong_signal(signals, pocket, "OPEN_LONG")

            pocket_fac = fac_h4 if pocket == "macro" else fac_m1
            rsi = pocket_fac.get("rsi", fac_m1.get("rsi", 50.0))
            ma10 = pocket_fac.get("ma10", 0.0)
            ma20 = pocket_fac.get("ma20", 0.0)
            adx = pocket_fac.get("adx", 0.0)
            ema20 = fac_m1.get("ema20", 0.0)
            close_price = fac_m1.get("close", 0.0)

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

        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                reason = "reverse_signal"
                allow_reentry = True
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
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if profit_pips >= 1.6:
                reason = "range_take_profit"
                allow_reentry = False
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"

        if range_mode and reason == "reverse_signal":
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

        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            macro_skip = (
                pocket == "macro"
                and reverse_signal.get("confidence", 0) < self._macro_signal_threshold
                and profit_pips > -self._macro_loss_buffer
            )
            if not macro_skip:
                reason = "reverse_signal"
                allow_reentry = True
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
        # レンジ中でもマクロの既存建玉を一律にクローズしない。
        # 早期利確/撤退（range_take_profit/range_stop）や逆方向シグナルのみで制御する。
        elif range_mode:
            if profit_pips >= 1.6:
                reason = "range_take_profit"
                allow_reentry = False
            elif profit_pips > 0.4:
                return None
            elif profit_pips <= -1.0:
                reason = "range_stop"

        if range_mode and reason == "reverse_signal":
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
