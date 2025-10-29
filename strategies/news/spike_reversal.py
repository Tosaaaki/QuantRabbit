from __future__ import annotations
from typing import Dict


class NewsSpikeReversal:
    name = "NewsSpikeReversal"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict, news_short: list[Dict]) -> Dict | None:
        # 直近のニュースと価格変動をチェック
        if not news_short:
            return None

        latest_news = news_short[0]
        # ニュースのセンチメントと価格の急変動を考慮
        # 例: 強い買いニュースで価格が急騰したら売り、強い売りニュースで価格が急落したら買い
        # ここでは簡略化のため、ニュースのセンチメントと直近のローソク足の動きを組み合わせる

        # ニュースのセンチメントを仮定 (例: sentiment: 1=positive, -1=negative, 0=neutral)
        # 実際にはGPTがニュースを要約する際にセンチメントを付与する想定
        news_sentiment = latest_news.get("sentiment", 0)  # 仮のセンチメント

        close_price = fac.get("close")
        open_price = fac.get("open")
        atr = fac.get("atr", 0.02)

        if close_price is None or open_price is None:
            return None

        price_change = close_price - open_price
        shock = abs(price_change) / max(0.001, atr)
        confidence = int(
            max(40.0, min(95.0, 50.0 + shock * 15.0 + abs(news_sentiment) * 15.0))
        )
        tag_suffix = "neg" if news_sentiment < 0 else "pos" if news_sentiment > 0 else "flat"

        # 強い買いニュースで価格が急騰したら売り (逆張り)
        if news_sentiment > 0 and price_change > 0.05:  # 0.05は仮の閾値
            return {
                "action": "OPEN_SHORT",
                "sl_pips": 18,
                "tp_pips": 6,
                "confidence": confidence,
                "tag": f"{NewsSpikeReversal.name}-fade-{tag_suffix}",
            }

        # 強い売りニュースで価格が急落したら買い (逆張り)
        if news_sentiment < 0 and price_change < -0.05:  # 0.05は仮の閾値
            return {
                "action": "OPEN_LONG",
                "sl_pips": 18,
                "tp_pips": 6,
                "confidence": confidence,
                "tag": f"{NewsSpikeReversal.name}-fade-{tag_suffix}",
            }

        return None
