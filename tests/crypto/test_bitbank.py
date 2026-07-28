from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from quant_rabbit.crypto.bitbank import (
    BitbankAPIError,
    BitbankPrivateReadOnlyClient,
    BitbankPublicClient,
)
from quant_rabbit.crypto.config import ScannerConfig


class Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_public_client_retries_429_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    def opener(request: object, timeout: float) -> Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://example.invalid",
                429,
                "rate limited",
                {"Retry-After": "0.01"},
                io.BytesIO(),
            )
        return Response({"success": 1, "data": {"tickers": []}})

    client = BitbankPublicClient(
        ScannerConfig(
            retry_attempts=2,
            min_request_interval_sec=0,
            retry_base_delay_sec=0,
        ),
        opener=opener,
        sleep=sleeps.append,
    )
    assert client.fetch_tickers_jpy() == []
    assert client.stats.rate_limits == 1
    assert client.stats.retries == 1
    assert sleeps == [0.01]


def test_public_client_fails_after_network_errors() -> None:
    def opener(request: object, timeout: float) -> Response:
        raise urllib.error.URLError("offline")

    client = BitbankPublicClient(
        ScannerConfig(
            retry_attempts=2,
            min_request_interval_sec=0,
            retry_base_delay_sec=0,
        ),
        opener=opener,
        sleep=lambda _: None,
    )
    with pytest.raises(BitbankAPIError):
        client.fetch_pair_settings()
    assert client.stats.retries == 1


def test_private_adapter_is_get_only_without_mutation_surface() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, timeout: float) -> Response:
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        return Response({"success": 1, "data": {"assets": [{"asset": "jpy"}]}})

    client = BitbankPrivateReadOnlyClient(
        "key",
        "secret",
        base_url="https://example.invalid",
        opener=opener,
        clock_ms=lambda: 123456789,
    )
    assert len(client.fetch_assets()) == 1
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/v1/user/assets")
    assert "Access-signature" in captured["headers"]
    assert hasattr(client, "fetch_margin_status")
    assert hasattr(client, "fetch_margin_positions")
    for forbidden in (
        "place_order",
        "cancel_order",
        "withdraw",
        "post",
        "request",
    ):
        assert not hasattr(client, forbidden)


def test_private_margin_status_is_get_only() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, timeout: float) -> Response:
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        return Response(
            {
                "success": 1,
                "data": {
                    "status": "NORMAL",
                    "buy_credit": "1000",
                    "sell_credit": "900",
                },
            }
        )

    status = BitbankPrivateReadOnlyClient(
        "key",
        "secret",
        base_url="https://example.invalid",
        opener=opener,
        clock_ms=lambda: 123456789,
    ).fetch_margin_status()
    assert status["status"] == "NORMAL"
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/v1/user/margin/status")
