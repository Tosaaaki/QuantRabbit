from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import CryptoSafetyContract, ScannerConfig


class BitbankAPIError(RuntimeError):
    pass


class BitbankRateLimitError(BitbankAPIError):
    pass


@dataclass(frozen=True)
class RequestStats:
    requests: int
    retries: int
    rate_limits: int


class BitbankPublicClient:
    """Dependency-free Public REST client with bounded 429/network backoff."""

    def __init__(
        self,
        config: ScannerConfig | None = None,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ScannerConfig.from_env()
        self._opener = opener
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at = 0.0
        self._requests = 0
        self._retries = 0
        self._rate_limits = 0

    @property
    def stats(self) -> RequestStats:
        return RequestStats(self._requests, self._retries, self._rate_limits)

    def _rate_limit(self) -> None:
        remaining = (
            self.config.min_request_interval_sec
            - (self._monotonic() - self._last_request_at)
        )
        if remaining > 0:
            self._sleep(remaining)

    def _request(self, url: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.retry_attempts):
            self._rate_limit()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "QuantRabbit-Crypto-ReadOnly/0.1",
                },
                method="GET",
            )
            self._requests += 1
            self._last_request_at = self._monotonic()
            try:
                with self._opener(
                    request, timeout=self.config.request_timeout_sec
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if payload.get("success") != 1:
                    raise BitbankAPIError(
                        f"bitbank public error code={payload.get('data', {}).get('code')}"
                    )
                return payload["data"]
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise BitbankAPIError(f"bitbank HTTP {exc.code}") from exc
                if exc.code == 429:
                    self._rate_limits += 1
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = self._backoff(attempt, retry_after)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                delay = self._backoff(attempt, None)
            if attempt + 1 < self.config.retry_attempts:
                self._retries += 1
                self._sleep(delay)
        if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
            raise BitbankRateLimitError("bitbank public API remained rate limited")
        raise BitbankAPIError(f"bitbank public request failed: {type(last_error).__name__}")

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        base = self.config.retry_base_delay_sec * (2**attempt)
        return min(30.0, base + random.uniform(0, base * 0.1))

    def fetch_pair_settings(self) -> list[dict[str, Any]]:
        data = self._request(f"{self.config.settings_base_url}/spot/pairs")
        return list(data.get("pairs", []))

    def fetch_exchange_status(self) -> list[dict[str, Any]]:
        data = self._request(f"{self.config.settings_base_url}/spot/status")
        return list(data.get("statuses", data.get("status", [])))

    def fetch_tickers_jpy(self) -> list[dict[str, Any]]:
        data = self._request(f"{self.config.public_base_url}/tickers_jpy")
        if isinstance(data, list):
            return data
        return list(data.get("tickers", []))

    def fetch_depth(self, pair: str) -> dict[str, Any]:
        return dict(self._request(f"{self.config.public_base_url}/{pair}/depth"))

    def fetch_circuit_break_info(self, pair: str) -> dict[str, Any]:
        return dict(
            self._request(
                f"{self.config.public_base_url}/{pair}/circuit_break_info"
            )
        )

    def fetch_candles(
        self, pair: str, candle_type: str, period: str
    ) -> list[list[Any]]:
        data = self._request(
            f"{self.config.public_base_url}/{pair}/candlestick/{candle_type}/{period}"
        )
        rows: list[list[Any]] = []
        for group in data.get("candlestick", []):
            if group.get("type") == candle_type:
                rows.extend(group.get("ohlcv", []))
        return rows


class BitbankPrivateReadOnlyClient:
    """Private GET-only client with no order, cancel, or withdrawal surface."""

    _ASSETS_PATH = "/v1/user/assets"
    _MARGIN_STATUS_PATH = "/v1/user/margin/status"
    _MARGIN_POSITIONS_PATH = "/v1/user/margin/positions"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://api.bitbank.cc",
        time_window_ms: int = 3000,
        timeout_sec: float = 8.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        CryptoSafetyContract.from_env().assert_safe()
        if not api_key or not api_secret:
            raise BitbankAPIError("bitbank read-only credentials are missing")
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._time_window_ms = time_window_ms
        self._timeout_sec = timeout_sec
        self._opener = opener
        self._clock_ms = clock_ms

    def fetch_assets(self) -> list[dict[str, Any]]:
        payload = self._get_private(self._ASSETS_PATH)
        return list(payload.get("assets", []))

    def fetch_margin_status(self) -> dict[str, Any]:
        return dict(self._get_private(self._MARGIN_STATUS_PATH))

    def fetch_margin_positions(self) -> dict[str, Any]:
        return dict(self._get_private(self._MARGIN_POSITIONS_PATH))

    def _get_private(self, path: str) -> dict[str, Any]:
        request_time = str(self._clock_ms())
        window = str(self._time_window_ms)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            f"{request_time}{window}{path}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            headers={
                "Accept": "application/json",
                "ACCESS-KEY": self._api_key,
                "ACCESS-REQUEST-TIME": request_time,
                "ACCESS-TIME-WINDOW": window,
                "ACCESS-SIGNATURE": signature,
                "User-Agent": "QuantRabbit-Crypto-Private-ReadOnly/0.1",
            },
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise BitbankAPIError(
                f"bitbank private read-only request failed: {type(exc).__name__}"
            ) from exc
        if payload.get("success") != 1:
            raise BitbankAPIError(
                f"bitbank private error code={payload.get('data', {}).get('code')}"
            )
        return dict(payload.get("data", {}))


def utc_from_ms(timestamp_ms: int | float | str) -> datetime:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc)
