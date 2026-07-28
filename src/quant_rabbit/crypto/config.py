from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal


class SafetyInvariantError(RuntimeError):
    """Raised when a crypto process is configured with live authority."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SafetyInvariantError(f"{name} must be an explicit boolean")


@dataclass(frozen=True)
class CryptoSafetyContract:
    no_execute: bool = True
    crypto_live_ready: bool = False
    withdrawal_enabled: bool = False
    order_authority: str = "NONE"

    @classmethod
    def from_env(cls) -> "CryptoSafetyContract":
        contract = cls(
            no_execute=_env_bool("NO_EXECUTE", True),
            crypto_live_ready=_env_bool("CRYPTO_LIVE_READY", False),
            withdrawal_enabled=_env_bool("WITHDRAWAL_ENABLED", False),
            order_authority=os.environ.get(
                "CRYPTO_ORDER_AUTHORITY", "NONE"
            ).strip().upper(),
        )
        contract.assert_safe()
        return contract

    def assert_safe(self) -> None:
        problems: list[str] = []
        if not self.no_execute:
            problems.append("NO_EXECUTE must remain true")
        if self.crypto_live_ready:
            problems.append("CRYPTO_LIVE_READY must remain false")
        if self.withdrawal_enabled:
            problems.append("WITHDRAWAL_ENABLED must remain false")
        if self.order_authority != "NONE":
            problems.append("CRYPTO_ORDER_AUTHORITY must remain NONE")
        if problems:
            raise SafetyInvariantError("; ".join(problems))

    def as_dict(self) -> dict[str, object]:
        return {
            "no_execute": self.no_execute,
            "crypto_live_ready": self.crypto_live_ready,
            "withdrawal_enabled": self.withdrawal_enabled,
            "order_authority": self.order_authority,
            "broker_mutation_allowed": False,
        }


@dataclass(frozen=True)
class ScannerConfig:
    public_base_url: str = "https://public.bitbank.cc"
    settings_base_url: str = "https://api.bitbank.cc/v1"
    stream_url: str = "wss://stream.bitbank.cc/socket.io/?EIO=4&transport=websocket"
    request_timeout_sec: float = 8.0
    min_request_interval_sec: float = 0.08
    retry_attempts: int = 4
    retry_base_delay_sec: float = 0.25
    max_ticker_age_sec: int = 120
    shortlist_size: int = 5
    detailed_pair_limit: int = 8
    min_volume_jpy: Decimal = Decimal("10000000")
    max_spread_bps: Decimal = Decimal("50")
    min_depth_25bps_jpy: Decimal = Decimal("500000")
    target_notional_jpy: Decimal = Decimal("10000")
    required_safety_buffer_bps: Decimal = Decimal("8")
    uncertainty_penalty_bps: Decimal = Decimal("5")

    @classmethod
    def from_env(cls) -> "ScannerConfig":
        return cls(
            public_base_url=os.environ.get(
                "QR_BITBANK_PUBLIC_BASE_URL", cls.public_base_url
            ),
            settings_base_url=os.environ.get(
                "QR_BITBANK_SETTINGS_BASE_URL", cls.settings_base_url
            ),
            stream_url=os.environ.get("QR_BITBANK_STREAM_URL", cls.stream_url),
            request_timeout_sec=float(
                os.environ.get("QR_BITBANK_REQUEST_TIMEOUT_SEC", cls.request_timeout_sec)
            ),
            min_request_interval_sec=float(
                os.environ.get(
                    "QR_BITBANK_MIN_REQUEST_INTERVAL_SEC", cls.min_request_interval_sec
                )
            ),
            retry_attempts=int(
                os.environ.get("QR_BITBANK_RETRY_ATTEMPTS", cls.retry_attempts)
            ),
            retry_base_delay_sec=float(
                os.environ.get(
                    "QR_BITBANK_RETRY_BASE_DELAY_SEC", cls.retry_base_delay_sec
                )
            ),
            max_ticker_age_sec=int(
                os.environ.get("QR_CRYPTO_MAX_TICKER_AGE_SEC", cls.max_ticker_age_sec)
            ),
            shortlist_size=int(
                os.environ.get("QR_CRYPTO_SHORTLIST_SIZE", cls.shortlist_size)
            ),
            detailed_pair_limit=int(
                os.environ.get(
                    "QR_CRYPTO_DETAILED_PAIR_LIMIT", cls.detailed_pair_limit
                )
            ),
            min_volume_jpy=Decimal(
                os.environ.get("QR_CRYPTO_MIN_VOLUME_JPY", str(cls.min_volume_jpy))
            ),
            max_spread_bps=Decimal(
                os.environ.get("QR_CRYPTO_MAX_SPREAD_BPS", str(cls.max_spread_bps))
            ),
            min_depth_25bps_jpy=Decimal(
                os.environ.get(
                    "QR_CRYPTO_MIN_DEPTH_25BPS_JPY", str(cls.min_depth_25bps_jpy)
                )
            ),
            target_notional_jpy=Decimal(
                os.environ.get(
                    "QR_CRYPTO_TARGET_NOTIONAL_JPY", str(cls.target_notional_jpy)
                )
            ),
            required_safety_buffer_bps=Decimal(
                os.environ.get(
                    "QR_CRYPTO_REQUIRED_BUFFER_BPS",
                    str(cls.required_safety_buffer_bps),
                )
            ),
            uncertainty_penalty_bps=Decimal(
                os.environ.get(
                    "QR_CRYPTO_UNCERTAINTY_PENALTY_BPS",
                    str(cls.uncertainty_penalty_bps),
                )
            ),
        )
