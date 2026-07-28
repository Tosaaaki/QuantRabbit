from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from quant_rabbit.crypto.bitbank import RequestStats
from quant_rabbit.crypto.config import ScannerConfig
from quant_rabbit.crypto.scanner import CryptoMarketScanner


class FakeClient:
    config = ScannerConfig(
        min_volume_jpy=Decimal("1000"),
        min_depth_25bps_jpy=Decimal("5000"),
        detailed_pair_limit=3,
        required_safety_buffer_bps=Decimal("5"),
    )
    stats = RequestStats(5, 0, 0)

    def __init__(self, timestamp_ms: int) -> None:
        self.timestamp_ms = timestamp_ms

    def fetch_pair_settings(self) -> list[dict[str, Any]]:
        base = {
            "quote_asset": "jpy",
            "is_enabled": True,
            "stop_order": False,
            "stop_order_and_cancel": False,
            "stop_buy_order": False,
            "maker_fee_rate_quote": "-0.0002",
            "taker_fee_rate_quote": "0.0012",
        }
        return [
            {"name": "btc_jpy", **base},
            {"name": "thin_jpy", **base},
            {"name": "stale_jpy", **base},
            {"name": "btc_usd", **{**base, "quote_asset": "usd"}},
        ]

    def fetch_exchange_status(self) -> list[dict[str, Any]]:
        return [
            {"pair": pair, "status": "NORMAL"}
            for pair in ("btc_jpy", "thin_jpy", "stale_jpy")
        ]

    def fetch_tickers_jpy(self) -> list[dict[str, Any]]:
        return [
            {
                "pair": "btc_jpy",
                "last": "110",
                "open": "100",
                "buy": "109.9",
                "sell": "110.1",
                "vol": "100000",
                "timestamp": self.timestamp_ms,
            },
            {
                "pair": "thin_jpy",
                "last": "110",
                "open": "100",
                "buy": "109",
                "sell": "111",
                "vol": "10000",
                "timestamp": self.timestamp_ms,
            },
            {
                "pair": "stale_jpy",
                "last": "110",
                "open": "100",
                "buy": "109.9",
                "sell": "110.1",
                "vol": "1000",
                "timestamp": self.timestamp_ms - 500_000,
            },
        ]

    def fetch_depth(self, pair: str) -> dict[str, Any]:
        amount = "10" if pair == "thin_jpy" else "10000"
        return {
            "asks": [["110.1", amount]],
            "bids": [["109.9", amount]],
        }

    def fetch_circuit_break_info(self, pair: str) -> dict[str, Any]:
        return {"mode": "NONE"}


def test_scanner_discovers_jpy_and_preserves_rejection_evidence() -> None:
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    client = FakeClient(int(now.timestamp() * 1000))
    result = CryptoMarketScanner(client).scan(now=now)
    assert result["counts"]["discovered_jpy_pairs"] == 3
    assert result["candidates"][0]["pair"] == "btc_jpy"
    by_pair = {item["pair"]: item for item in result["pairs"]}
    assert "THIN_BOOK" in by_pair["thin_jpy"]["reasons"]
    assert "STALE_TICKER" in by_pair["stale_jpy"]["reasons"]
    assert result["guardian"]["state"] == "GREEN"
    assert result["virtual_intents"][0]["authority"] == "NONE"
    assert result["virtual_intents"][0]["live_permission"] is False


def test_all_stale_data_deterministically_halts() -> None:
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    stale = now - timedelta(hours=1)
    result = CryptoMarketScanner(
        FakeClient(int(stale.timestamp() * 1000))
    ).scan(now=now)
    assert result["guardian"]["state"] == "HALT"
    assert result["guardian"]["kill_switch"] is True
    assert result["virtual_intents"] == []
