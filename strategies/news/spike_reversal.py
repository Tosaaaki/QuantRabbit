from __future__ import annotations
from typing import Dict
from datetime import datetime, timezone


class NewsSpikeReversal:
    name = "NewsSpikeReversal"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict, news_short: list[Dict]) -> Dict | None:
        # 直近のニュースと価格変動をチェック
        if not news_short:
            return None

        # USD/JPY 関連のニュースを優先（pair_bias に USD_JPY が含まれるもの）
        latest = None
        for item in news_short:
            pair_bias = (item.get("pair_bias") or "").upper()
            if "USD_JPY" in pair_bias or pair_bias in ("USDJPY", "USD-JPY"):
                latest = item
                break
        if latest is None:
            latest = news_short[0]
        # 影響度フィルタ（中〜高インパクト）+ 新鮮さ（発表から20分以内）
        impact = int(latest.get("impact", 1) or 1)
        if impact < 2:
            return None
        evt_iso = latest.get("event_time") or latest.get("ts")
        try:
            evt_ts = datetime.fromisoformat(evt_iso.replace("Z", "+00:00")).astimezone(timezone.utc)  # type: ignore[arg-type]
        except Exception:
            evt_ts = None
        if evt_ts is None:
            return None
        now = datetime.now(timezone.utc)
        if (now - evt_ts).total_seconds() > 20 * 60:
            return None

        news_sentiment = int(latest.get("sentiment", 0) or 0)

        close_price = fac.get("close")
        open_price = fac.get("open")
        if close_price is None or open_price is None:
            return None
        price_change = float(close_price) - float(open_price)

        # ATRベースで閾値自動調整（USD/JPY: 1pip=0.01）
        atr = float(fac.get("atr", 0.0) or 0.0)
        atr_pips = max(atr * 100.0, 0.0)
        # 目安: min=5pips, max=15pips、通常は ATR の 0.8 倍
        dyn_th_pips = max(5.0, min(15.0, atr_pips * 0.8))
        th = dyn_th_pips / 100.0  # 価格単位へ戻す

        # 強い買いニュースで価格が急騰したら売り (逆張り)
        if news_sentiment > 0 and price_change > th:
            sl = int(max(8, min(20, atr_pips * 1.2)))
            tp = int(sl * 2)
            return {"action": "sell", "sl_pips": sl, "tp_pips": tp, "ttl_sec": 900}

        # 強い売りニュースで価格が急落したら買い (逆張り)
        if news_sentiment < 0 and price_change < -th:
            sl = int(max(8, min(20, atr_pips * 1.2)))
            tp = int(sl * 2)
            return {"action": "buy", "sl_pips": sl, "tp_pips": tp, "ttl_sec": 900}

        return None
