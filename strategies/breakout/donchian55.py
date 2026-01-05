import pandas as pd
from typing import Dict
import logging
from analysis.ma_projection import compute_donchian_projection


class Donchian55:
    name = "Donchian55"
    pocket = "macro"
    profile = "macro_breakout_donchian"

    @staticmethod
    def _log_skip(reason: str, **kwargs) -> None:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        logging.info("[STRAT_SKIP_DETAIL] Donchian55 reason=%s %s", reason, extras)

    @staticmethod
    def check(fac: Dict, *, range_active: bool = False) -> Dict | None:
        candles = fac.get("candles")
        if candles is None or len(candles) < 56:
            Donchian55._log_skip("insufficient_candles", count=len(candles) if candles is not None else 0)
            return None

        df = pd.DataFrame(candles)[-56:]
        high55 = df["high"][:-1].max()
        low55 = df["low"][:-1].min()
        close = df["close"].iloc[-1]
        if any(val is None for val in (high55, low55, close)):
            Donchian55._log_skip("nan_values", high55=high55, low55=low55, close=close)
            return None
        range_span = max(1e-6, high55 - low55)
        try:
            spread_pips = float(fac.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = 0.0
        breakout_strength = abs(close - (high55 + low55) / 2) / range_span
        try:
            adx = float(fac.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            atr_pips = float(fac.get("atr_pips") or 0.0)
        except (TypeError, ValueError):
            atr_pips = 0.0
        try:
            rsi = float(fac.get("rsi") or 0.0)
        except (TypeError, ValueError):
            rsi = 50.0
        # 勢いフィルター（トレンド弱・RSI中立ならスキップ）
        if adx < 22.0:
            Donchian55._log_skip("weak_adx", adx=round(adx, 2))
            return None
        if close >= high55 and rsi < 52.0:
            Donchian55._log_skip("weak_momentum_long", rsi=round(rsi, 2))
            return None
        if close <= low55 and rsi > 48.0:
            Donchian55._log_skip("weak_momentum_short", rsi=round(rsi, 2))
            return None

        # ブレイクへの近さ（pips）で発火加減を調整
        proj = compute_donchian_projection(candles, lookback=55)
        near_pips = proj.nearest_pips if proj else None
        distance_bonus = 0.0
        if near_pips is not None:
            distance_bonus = max(0.0, min(10.0, (8.0 - min(8.0, near_pips)) * 1.2))
        confidence = int(max(45.0, min(95.0, 54.0 + breakout_strength * 42.0 + distance_bonus)))

        # レンジ／低勢い時はそもそもエントリー抑制
        if range_active and breakout_strength < 0.4:
            Donchian55._log_skip(
                "range_block",
                breakout_strength=round(breakout_strength, 3),
                near_pips=round(near_pips, 2) if near_pips is not None else None,
            )
            return None

        def _targets() -> tuple[float, float]:
            base_sl = 55.0
            base_tp = 110.0
            spread_floor = max(20.0, spread_pips * 2.5 + 12.0)
            sl = max(base_sl, spread_floor)
            # TP をチャート形状に合わせて縮小/拡張
            tp = max(base_tp, sl * 1.35 + spread_pips)
            vol_scale = 1.0
            if atr_pips < 4.0:
                vol_scale *= 0.65
            elif atr_pips < 6.0:
                vol_scale *= 0.85
            elif atr_pips > 9.0:
                vol_scale *= 1.12
            trend_scale = 0.75 if adx < 18.0 else 1.0 if adx < 32.0 else 1.12
            range_scale = 0.55 if range_active else 1.0
            momentum_scale = 0.95 + min(0.35, breakout_strength * 0.35)
            tp *= max(0.45, min(1.35, vol_scale * trend_scale * range_scale * momentum_scale))
            # 到達確度を優先し、ATRと55本レンジで上限を強くクランプ
            range_pips = max(10.0, range_span * 100.0)  # 55本レンジをpip換算
            tp_cap = min(tp, 24.0, atr_pips * 4.5 + 6.0, range_pips * 0.35)
            tp_floor = max(10.0, sl * 0.5, atr_pips * 2.5)
            tp = max(tp_floor, min(tp_cap, sl * 1.25))
            return round(sl, 2), round(tp, 2)

        if close > high55 or (near_pips is not None and near_pips <= 3.0 and close >= high55 - 0.02):
            sl, tp = _targets()
            return {
                "action": "OPEN_LONG",
                "sl_pips": sl,
                "tp_pips": tp,
                "confidence": confidence,
                "profile": Donchian55.profile,
                "loss_guard_pips": round(sl * 0.7, 2),
                "target_tp_pips": tp,
                "min_hold_sec": Donchian55._min_hold_seconds(tp),
                "tag": f"{Donchian55.name}-breakout-up",
            }
        if close < low55 or (near_pips is not None and near_pips <= 3.0 and close <= low55 + 0.02):
            sl, tp = _targets()
            return {
                "action": "OPEN_SHORT",
                "sl_pips": sl,
                "tp_pips": tp,
                "confidence": confidence,
                "profile": Donchian55.profile,
                "loss_guard_pips": round(sl * 0.7, 2),
                "target_tp_pips": tp,
                "min_hold_sec": Donchian55._min_hold_seconds(tp),
                "tag": f"{Donchian55.name}-breakout-down",
            }
        Donchian55._log_skip(
            "no_breakout",
            close=close,
            high55=high55,
            low55=low55,
            near_pips=round(near_pips, 3) if near_pips is not None else None,
        )
        return None

    @staticmethod
    def _min_hold_seconds(tp_pips: float) -> float:
        return round(max(420.0, min(1800.0, tp_pips * 8.5)), 1)
