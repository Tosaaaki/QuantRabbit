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

    def plan_closures(
        self,
        open_positions: Dict[str, Dict],
        signals: List[Dict],
        fac_m1: Dict,
        fac_h4: Dict,
        event_soon: bool,
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
                    long_units,
                    reverse_short,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema20,
                )
                if decision:
                    decisions.append(decision)

            if short_units > 0:
                decision = self._evaluate_short(
                    pocket,
                    short_units,
                    reverse_long,
                    event_soon,
                    rsi,
                    ma10,
                    ma20,
                    adx,
                    close_price,
                    ema20,
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
        if best.get("confidence", 0) >= self.confidence_threshold:
            return best
        return None

    def _evaluate_long(
        self,
        pocket: str,
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema20: float,
    ) -> Optional[ExitDecision]:
        allow_reentry = False
        reason = ""
        tag = f"{pocket}-long"

        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            reason = "reverse_signal"
            allow_reentry = True
            tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi >= 65:
            reason = "rsi_overbought"
        elif pocket == "macro" and ma10 < ma20 and adx <= 18:
            reason = "trend_reversal"
        elif pocket == "scalp" and close_price > ema20:
            reason = "scalp_momentum_flip"

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
        units: int,
        reverse_signal: Optional[Dict],
        event_soon: bool,
        rsi: float,
        ma10: float,
        ma20: float,
        adx: float,
        close_price: float,
        ema20: float,
    ) -> Optional[ExitDecision]:
        allow_reentry = False
        reason = ""
        tag = f"{pocket}-short"

        if event_soon and pocket in {"micro", "scalp"}:
            reason = "event_lock"
        elif reverse_signal:
            reason = "reverse_signal"
            allow_reentry = True
            tag = reverse_signal.get("tag", tag)
        elif pocket == "micro" and rsi <= 35:
            reason = "rsi_oversold"
        elif pocket == "macro" and ma10 > ma20 and adx <= 18:
            reason = "trend_reversal"
        elif pocket == "scalp" and close_price < ema20:
            reason = "scalp_momentum_flip"

        if not reason:
            return None

        return ExitDecision(
            pocket=pocket,
            units=abs(units),
            reason=reason,
            tag=tag,
            allow_reentry=allow_reentry,
        )
