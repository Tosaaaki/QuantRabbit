"""Offline tests for currency strength + options skew adapter."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.analysis.candles import Candle
from quant_rabbit.analysis.options import build_option_skew_snapshot
from quant_rabbit.analysis.strength import build_strength_snapshot, G8_CURRENCIES


class _StubClient:
    """Minimal client stub returning preset candle payloads keyed by pair."""

    def __init__(self, payloads: dict[str, list[Candle]]) -> None:
        self.payloads = payloads
        self.account_id = "stub"
        self.base_url = ""
        self.token = ""

    def get_json(self, path: str, query: dict[str, str] | None = None) -> dict:
        # used by fetch_candles when called against this stub
        # Identify pair from path: /v3/instruments/<pair>/candles
        parts = path.split("/")
        pair = parts[3]
        candles = self.payloads.get(pair)
        if candles is None:
            raise RuntimeError(f"no payload for {pair}")
        return {"candles": [
            {
                "time": c.timestamp_utc.isoformat().replace("+00:00", "Z"),
                "complete": True,
                "volume": c.volume,
                "mid": {"o": str(c.open), "h": str(c.high), "l": str(c.low), "c": str(c.close)},
            }
            for c in candles
        ]}


def _make(closes: list[float]) -> list[Candle]:
    base = datetime(2026, 5, 4, tzinfo=timezone.utc)
    return [
        Candle(
            timestamp_utc=base + timedelta(hours=i),
            open=c, high=c + 0.1, low=c - 0.1, close=c, volume=1000, complete=True,
        )
        for i, c in enumerate(closes)
    ]


class StrengthSnapshotTest(unittest.TestCase):
    def test_strength_ranks_strongest_currency_first(self) -> None:
        # USD strengthens vs everything: USD/* rises, */USD falls
        usd_up = _make([100.0 + i * 0.02 for i in range(50)])
        eur_down = _make([1.10 - i * 0.0005 for i in range(50)])  # EUR/USD down → USD up
        gbp_down = _make([1.25 - i * 0.0005 for i in range(50)])  # GBP/USD down → USD up
        usd_jpy_up = _make([155.0 + i * 0.02 for i in range(50)])  # USD/JPY up → USD up vs JPY
        # Provide other pairs as flat (no impact)
        flat_eur_jpy = _make([170.0 for _ in range(50)])
        payloads = {
            "EUR_USD": eur_down, "GBP_USD": gbp_down, "USD_JPY": usd_jpy_up,
            "AUD_USD": _make([0.65 - i * 0.0005 for i in range(50)]),
            "NZD_USD": _make([0.60 - i * 0.0005 for i in range(50)]),
            "USD_CAD": _make([1.35 + i * 0.0005 for i in range(50)]),
            "USD_CHF": _make([0.90 + i * 0.0005 for i in range(50)]),
            "EUR_GBP": flat_eur_jpy,
            "EUR_JPY": flat_eur_jpy, "EUR_AUD": flat_eur_jpy, "EUR_CAD": flat_eur_jpy,
            "EUR_CHF": flat_eur_jpy, "EUR_NZD": flat_eur_jpy,
            "GBP_JPY": flat_eur_jpy, "GBP_AUD": flat_eur_jpy, "GBP_CAD": flat_eur_jpy,
            "GBP_CHF": flat_eur_jpy, "GBP_NZD": flat_eur_jpy,
            "AUD_JPY": flat_eur_jpy, "AUD_CAD": flat_eur_jpy, "AUD_CHF": flat_eur_jpy,
            "AUD_NZD": flat_eur_jpy, "CAD_JPY": flat_eur_jpy, "CAD_CHF": flat_eur_jpy,
            "CHF_JPY": flat_eur_jpy, "NZD_JPY": flat_eur_jpy, "NZD_CAD": flat_eur_jpy,
            "NZD_CHF": flat_eur_jpy,
        }
        client = _StubClient(payloads)
        snap = build_strength_snapshot(client=client, lookback_bars=24, fetch_count=50)
        rank_by_currency = {s.currency: s.rank for s in snap.scores}
        self.assertIn("USD", rank_by_currency)
        # USD should be at or near rank 1 in this construction
        self.assertLessEqual(rank_by_currency["USD"], 2)
        self.assertEqual(set(rank_by_currency.keys()), set(G8_CURRENCIES))


class OptionSkewAdapterTest(unittest.TestCase):
    def test_default_emits_missing_feed_issue(self) -> None:
        snap = build_option_skew_snapshot(pairs=("USD_JPY",), tenors=("1W", "1M"))
        self.assertTrue(any("MISSING_OPTION_SKEW_FEED" in i for i in snap.issues))
        # All readings carry the missing-feed marker
        for r in snap.readings:
            self.assertIsNone(r.atm_iv)
            self.assertEqual(r.issue, "MISSING_OPTION_SKEW_FEED")


if __name__ == "__main__":
    unittest.main()
