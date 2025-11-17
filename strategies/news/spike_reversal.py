from __future__ import annotations
from typing import Dict


class NewsSpikeReversal:
    name = "NewsSpikeReversal"
    pocket = "micro"
    profile = "micro_news_reversal"

    _MIN_SHOCK = 1.4
    _TREND_ADX_GUARD = 28.0
    _MA_OFFSET_PIPS = 0.02  # ≒2pips

    @staticmethod
    def _float(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _targets(atr_pips: float, spread_pips: float) -> tuple[float, float]:
        atr_pips = max(6.0, atr_pips)
        sl = max(12.0, atr_pips * 1.6, spread_pips * 2.8 + 8.0)
        tp = max(sl * 1.45, sl + atr_pips * 0.9) + spread_pips * 0.8
        return round(sl, 2), round(tp, 2)

    @staticmethod
    def check(fac: Dict, news_short: list[Dict]) -> Dict | None:
        if not news_short:
            return None
        latest_news = news_short[0]
        news_sentiment = NewsSpikeReversal._float(latest_news.get("sentiment"), 0.0)

        close_price = fac.get("close")
        open_price = fac.get("open")
        if close_price is None or open_price is None:
            return None

        atr_pips = NewsSpikeReversal._float(
            fac.get("atr_pips"), NewsSpikeReversal._float(fac.get("atr"), 0.02) * 100
        )
        ema20 = fac.get("ema20") or fac.get("ma20")
        adx = NewsSpikeReversal._float(fac.get("adx"), 0.0)
        spread_pips = NewsSpikeReversal._float(fac.get("spread_pips"), 0.0)

        price_change = NewsSpikeReversal._float(close_price) - NewsSpikeReversal._float(open_price)
        atr_base = max(0.01, NewsSpikeReversal._float(fac.get("atr"), 0.02))
        shock = abs(price_change) / atr_base
        if shock < NewsSpikeReversal._MIN_SHOCK or abs(news_sentiment) < 0.5:
            return None

        ema_guard = ema20 is not None and (
            (news_sentiment > 0 and close_price > ema20 + NewsSpikeReversal._MA_OFFSET_PIPS)
            or (news_sentiment < 0 and close_price < ema20 - NewsSpikeReversal._MA_OFFSET_PIPS)
        )
        strong_trend = adx >= NewsSpikeReversal._TREND_ADX_GUARD
        if strong_trend and (price_change * news_sentiment) > 0:
            return None
        if ema_guard:
            return None

        sl_pips, tp_pips = NewsSpikeReversal._targets(atr_pips, spread_pips)
        confidence = int(
            max(
                40.0,
                min(
                    95.0,
                    52.0
                    + (shock - NewsSpikeReversal._MIN_SHOCK) * 12.0
                    + abs(news_sentiment) * 12.0
                    - (5.0 if strong_trend else 0.0),
                ),
            )
        )
        tag_suffix = "neg" if news_sentiment < 0 else "pos"

        if news_sentiment > 0 and price_change > 0.0:
            min_hold = NewsSpikeReversal._min_hold_seconds(tp_pips)
            loss_guard = round(max(2.5, min(sl_pips * 0.7, tp_pips * 0.5)), 2)
            return {
                "action": "OPEN_SHORT",
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "confidence": confidence,
                "profile": NewsSpikeReversal.profile,
                "target_tp_pips": tp_pips,
                "loss_guard_pips": loss_guard,
                "min_hold_sec": min_hold,
                "tag": f"{NewsSpikeReversal.name}-fade-{tag_suffix}",
            }
        if news_sentiment < 0 and price_change < 0.0:
            min_hold = NewsSpikeReversal._min_hold_seconds(tp_pips)
            loss_guard = round(max(2.5, min(sl_pips * 0.7, tp_pips * 0.5)), 2)
            return {
                "action": "OPEN_LONG",
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "confidence": confidence,
                "profile": NewsSpikeReversal.profile,
                "target_tp_pips": tp_pips,
                "loss_guard_pips": loss_guard,
                "min_hold_sec": min_hold,
                "tag": f"{NewsSpikeReversal.name}-fade-{tag_suffix}",
            }
        return None

    @staticmethod
    def _min_hold_seconds(tp_pips: float) -> float:
        return round(max(60.0, min(300.0, tp_pips * 35.0)), 1)
