"""Bounded OANDA read-only adapters for future paper AI inventory evidence.

The adapters in this module have no caller-supplied URL, path, client, or
transport hook.  They instantiate :class:`OandaReadOnlyClient` and call only
its current-quote or fixed candle-read surfaces.  Their strict, content-
addressed configuration is intended to be stored under the source-capture
adapter-config root and bound by a lifecycle-approved capture manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


SOURCE_ADAPTER_CONFIG_CONTRACT = "QR_DOJO_AI_SOURCE_ADAPTER_CONFIG_V1"
OANDA_EXECUTABLE_QUOTE_ADAPTER_ID = "oanda-executable-quote-v1"
OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID = "oanda-completed-bid-ask-candles-v1"
OANDA_READ_ONLY_PROVIDER_KIND = "OANDA_V20_READ_ONLY"
SOURCE_ADAPTER_MODULE = "quant_rabbit.dojo_ai_source_adapters"

OANDA_PAIR_ALLOWLIST = frozenset(DEFAULT_TRADER_PAIRS)
OANDA_CANDLE_GRANULARITY_SECONDS = {
    "S5": 5,
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
}
MAX_OANDA_CANDLE_COUNT = 5_000
MAX_SOURCE_ADAPTER_BYTES = 4 * 1024 * 1024
MAX_QUOTE_AGE_SECONDS = 180
MAX_CANDLE_AGE_SECONDS = 24 * 60 * 60

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMON_CONFIG_KEYS = frozenset(
    {
        "contract",
        "adapter_id",
        "pair",
        "max_age_seconds",
        "paper_only",
        "order_authority",
        "live_permission",
        "config_sha256",
    }
)
_CANDLE_CONFIG_KEYS = _COMMON_CONFIG_KEYS | frozenset(
    {
        "granularity",
        "count",
        "price_component",
        "smooth",
        "complete_only",
    }
)
_QUOTE_OUTPUT_KEYS = frozenset(
    {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "max_age_seconds",
    }
)
_CANDLE_KEYS = frozenset(
    {
        "pair",
        "granularity",
        "started_at_utc",
        "completed_at_utc",
        "bid_o",
        "bid_h",
        "bid_l",
        "bid_c",
        "ask_o",
        "ask_h",
        "ask_l",
        "ask_c",
        "max_age_seconds",
    }
)


class DojoAiSourceAdapterError(RuntimeError):
    """A source configuration or provider response failed closed."""


class DojoAiSourceAdapterMarketClosedError(DojoAiSourceAdapterError):
    """Network source evaluation is disabled outside the FX week."""


@dataclass(frozen=True)
class SourceAdapterAcquisition:
    """One immutable result accepted by the source-capture registration layer."""

    raw_bytes: bytes
    provider_timestamp_utc: str
    source_watermark_sha256: str


def source_adapter_config_sha256(value: Mapping[str, Any]) -> str:
    """Return the semantic digest of a strict config body."""

    return _normalize_config(value, require_digest=False)["config_sha256"]


def seal_source_adapter_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one strict config with its semantic digest attached."""

    return _normalize_config(value, require_digest=False)


def canonical_source_adapter_config_bytes(value: Mapping[str, Any]) -> bytes:
    """Return canonical newline-terminated bytes for the fixed config root."""

    normalized = _normalize_config(
        value,
        require_digest="config_sha256" in value,
    )
    return _canonical_bytes(normalized) + b"\n"


def source_adapter_module_sha256() -> str:
    """Hash the exact module bytes independently bound by capture manifests."""

    path = Path(__file__).resolve(strict=True)
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_SOURCE_ADAPTER_BYTES:
        raise DojoAiSourceAdapterError("source adapter module size is invalid")
    return hashlib.sha256(raw).hexdigest()


def source_adapter_capture_binding(value: Mapping[str, Any]) -> dict[str, str]:
    """Build the exact adapter row shape consumed by a capture manifest.

    The capture layer must independently hash the module and stored config
    bytes.  This helper supplies deterministic registration material; it is not
    a substitute for that independent verification.
    """

    config = _normalize_config(
        value,
        require_digest="config_sha256" in value,
    )
    if config["adapter_id"] == OANDA_EXECUTABLE_QUOTE_ADAPTER_ID:
        source_role = "quote"
        callable_name = "acquire_oanda_executable_quote"
    else:
        source_role = "candles"
        callable_name = "acquire_oanda_completed_bid_ask_candles"
    return {
        "source_role": source_role,
        "provider_kind": OANDA_READ_ONLY_PROVIDER_KIND,
        "adapter_id": config["adapter_id"],
        "adapter_module": SOURCE_ADAPTER_MODULE,
        "adapter_callable": callable_name,
        "adapter_executable_sha256": source_adapter_module_sha256(),
        "adapter_config_sha256": hashlib.sha256(
            canonical_source_adapter_config_bytes(config)
        ).hexdigest(),
    }


def acquire_oanda_executable_quote(
    config_value: Mapping[str, Any],
) -> SourceAdapterAcquisition:
    """Acquire one current executable bid/ask quote through read-only pricing."""

    pre_fetch = _utc_now()
    _require_market_open(pre_fetch)
    config = _normalize_config(config_value, require_digest=True)
    if config["adapter_id"] != OANDA_EXECUTABLE_QUOTE_ADAPTER_ID:
        raise DojoAiSourceAdapterError("quote adapter config identity mismatch")

    client = OandaReadOnlyClient()
    quotes = client.quotes((config["pair"],))
    fetched_at = _utc_now()
    _require_market_open(fetched_at)
    if not isinstance(quotes, Mapping) or set(quotes) != {config["pair"]}:
        raise DojoAiSourceAdapterError(
            "OANDA pricing response does not contain exactly the configured pair"
        )
    quote = quotes[config["pair"]]
    if getattr(quote, "pair", None) != config["pair"]:
        raise DojoAiSourceAdapterError("OANDA quote pair binding mismatch")
    bid = _positive_number(getattr(quote, "bid", None), "quote bid")
    ask = _positive_number(getattr(quote, "ask", None), "quote ask")
    if ask < bid:
        raise DojoAiSourceAdapterError("OANDA executable quote is crossed")
    provider_at = _utc(
        getattr(quote, "timestamp_utc", None),
        "quote provider timestamp",
    )
    _require_provider_time(
        provider_at,
        fetched_at=fetched_at,
        max_age_seconds=config["max_age_seconds"],
    )
    body: dict[str, Any] = {
        "pair": config["pair"],
        "bid": bid,
        "ask": ask,
        "timestamp_utc": _format_utc(provider_at),
        "max_age_seconds": config["max_age_seconds"],
    }
    if set(body) != _QUOTE_OUTPUT_KEYS:
        raise AssertionError("internal quote source schema mismatch")
    return _acquisition(body, provider_timestamp_utc=provider_at)


def acquire_oanda_completed_bid_ask_candles(
    config_value: Mapping[str, Any],
) -> SourceAdapterAcquisition:
    """Acquire an exact completed BID/ASK candle batch from a fixed endpoint."""

    pre_fetch = _utc_now()
    _require_market_open(pre_fetch)
    config = _normalize_config(config_value, require_digest=True)
    if config["adapter_id"] != OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID:
        raise DojoAiSourceAdapterError("candle adapter config identity mismatch")

    client = OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{config['pair']}/candles",
        {
            "granularity": config["granularity"],
            "count": str(config["count"]),
            "price": "BA",
            "smooth": "false",
        },
    )
    fetched_at = _utc_now()
    _require_market_open(fetched_at)
    candles = _completed_bid_ask_candles(
        payload,
        config=config,
        fetched_at=fetched_at,
    )
    latest = _utc(candles[-1]["completed_at_utc"], "latest completed candle")
    _require_provider_time(
        latest,
        fetched_at=fetched_at,
        max_age_seconds=config["max_age_seconds"],
    )
    return _acquisition(candles, provider_timestamp_utc=latest)


def _normalize_config(
    value: Mapping[str, Any],
    *,
    require_digest: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAiSourceAdapterError("source adapter config must be a mapping")
    try:
        snapshot = json.loads(
            _canonical_bytes(value),
            object_pairs_hook=_strict_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DojoAiSourceAdapterError(
            "source adapter config is not strict JSON"
        ) from exc
    adapter_id = snapshot.get("adapter_id")
    if adapter_id == OANDA_EXECUTABLE_QUOTE_ADAPTER_ID:
        expected = _COMMON_CONFIG_KEYS
    elif adapter_id == OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID:
        expected = _CANDLE_CONFIG_KEYS
    else:
        raise DojoAiSourceAdapterError("source adapter is not allowlisted")
    if not require_digest:
        expected = expected - {"config_sha256"}
    if set(snapshot) != expected:
        raise DojoAiSourceAdapterError("source adapter config schema is invalid")
    if snapshot.get("contract") != SOURCE_ADAPTER_CONFIG_CONTRACT:
        raise DojoAiSourceAdapterError("source adapter config contract is invalid")
    if (
        snapshot.get("paper_only") is not True
        or snapshot.get("order_authority") != "NONE"
        or snapshot.get("live_permission") is not False
    ):
        raise DojoAiSourceAdapterError(
            "source adapter config safety authority is invalid"
        )
    pair = snapshot.get("pair")
    if not isinstance(pair, str) or pair not in OANDA_PAIR_ALLOWLIST:
        raise DojoAiSourceAdapterError("source adapter pair is not allowlisted")
    max_age = _exact_int(
        snapshot.get("max_age_seconds"),
        "max_age_seconds",
        minimum=1,
        maximum=(
            MAX_QUOTE_AGE_SECONDS
            if adapter_id == OANDA_EXECUTABLE_QUOTE_ADAPTER_ID
            else MAX_CANDLE_AGE_SECONDS
        ),
    )
    normalized: dict[str, Any] = {
        "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
        "adapter_id": adapter_id,
        "pair": pair,
        "max_age_seconds": max_age,
    }
    if adapter_id == OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID:
        granularity = snapshot.get("granularity")
        if granularity not in OANDA_CANDLE_GRANULARITY_SECONDS:
            raise DojoAiSourceAdapterError(
                "source adapter candle granularity is not allowlisted"
            )
        if (
            snapshot.get("price_component") != "BA"
            or snapshot.get("smooth") is not False
            or snapshot.get("complete_only") is not True
        ):
            raise DojoAiSourceAdapterError(
                "source adapter candle mode must be exact completed BID/ASK"
            )
        normalized.update(
            {
                "granularity": granularity,
                "count": _exact_int(
                    snapshot.get("count"),
                    "count",
                    minimum=1,
                    maximum=MAX_OANDA_CANDLE_COUNT,
                ),
                "price_component": "BA",
                "smooth": False,
                "complete_only": True,
            }
        )
    normalized.update(
        {
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
    )
    digest = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    if require_digest and snapshot.get("config_sha256") != digest:
        raise DojoAiSourceAdapterError("source adapter config digest mismatch")
    normalized["config_sha256"] = digest
    return normalized


def _completed_bid_ask_candles(
    payload: object,
    *,
    config: Mapping[str, Any],
    fetched_at: datetime,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "instrument",
        "granularity",
        "candles",
    }:
        raise DojoAiSourceAdapterError("OANDA candle response schema is invalid")
    if (
        payload.get("instrument") != config["pair"]
        or payload.get("granularity") != config["granularity"]
    ):
        raise DojoAiSourceAdapterError("OANDA candle response binding mismatch")
    rows = payload.get("candles")
    if not isinstance(rows, list) or len(rows) != config["count"]:
        raise DojoAiSourceAdapterError(
            "OANDA candle response count does not match configured count"
        )
    duration = timedelta(
        seconds=OANDA_CANDLE_GRANULARITY_SECONDS[config["granularity"]]
    )
    normalized: list[dict[str, Any]] = []
    previous: datetime | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "time",
            "volume",
            "complete",
            "bid",
            "ask",
        }:
            raise DojoAiSourceAdapterError(f"OANDA candle[{index}] schema is invalid")
        if row.get("complete") is not True:
            raise DojoAiSourceAdapterError(f"OANDA candle[{index}] is incomplete")
        started = _utc(row.get("time"), f"OANDA candle[{index}] time")
        completed = started + duration
        if completed > fetched_at:
            raise DojoAiSourceAdapterError(f"OANDA candle[{index}] is future-dated")
        if previous is not None and started <= previous:
            raise DojoAiSourceAdapterError(
                "OANDA candle timestamps are not strictly increasing"
            )
        previous = started
        bid = _ohlc(row.get("bid"), f"OANDA candle[{index}] bid")
        ask = _ohlc(row.get("ask"), f"OANDA candle[{index}] ask")
        for component in ("o", "h", "l", "c"):
            if ask[component] < bid[component]:
                raise DojoAiSourceAdapterError(
                    f"OANDA candle[{index}] has crossed BID/ASK {component}"
                )
        candle = {
            "pair": config["pair"],
            "granularity": config["granularity"],
            "started_at_utc": _format_utc(started),
            "completed_at_utc": _format_utc(completed),
            "bid_o": bid["o"],
            "bid_h": bid["h"],
            "bid_l": bid["l"],
            "bid_c": bid["c"],
            "ask_o": ask["o"],
            "ask_h": ask["h"],
            "ask_l": ask["l"],
            "ask_c": ask["c"],
            "max_age_seconds": config["max_age_seconds"],
        }
        _exact_int(
            row.get("volume"),
            f"OANDA candle[{index}] volume",
            minimum=0,
            maximum=2**63 - 1,
        )
        if set(candle) != _CANDLE_KEYS:
            raise AssertionError("internal candle schema mismatch")
        normalized.append(candle)
    return normalized


def _ohlc(value: object, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"o", "h", "l", "c"}:
        raise DojoAiSourceAdapterError(f"{field} OHLC schema is invalid")
    result = {
        component: _positive_number(value.get(component), f"{field}.{component}")
        for component in ("o", "h", "l", "c")
    }
    if result["h"] < max(result["o"], result["l"], result["c"]) or result["l"] > min(
        result["o"], result["h"], result["c"]
    ):
        raise DojoAiSourceAdapterError(f"{field} OHLC geometry is invalid")
    return result


def _acquisition(
    value: Mapping[str, Any] | list[dict[str, Any]],
    *,
    provider_timestamp_utc: datetime,
) -> SourceAdapterAcquisition:
    raw = _canonical_bytes(value) + b"\n"
    if len(raw) > MAX_SOURCE_ADAPTER_BYTES:
        raise DojoAiSourceAdapterError("source adapter result is too large")
    watermark = hashlib.sha256(raw).hexdigest()
    return SourceAdapterAcquisition(
        raw_bytes=raw,
        provider_timestamp_utc=_format_utc(provider_timestamp_utc),
        source_watermark_sha256=watermark,
    )


def _require_provider_time(
    provider_at: datetime,
    *,
    fetched_at: datetime,
    max_age_seconds: int,
) -> None:
    if provider_at > fetched_at:
        raise DojoAiSourceAdapterError("provider timestamp is future-dated")
    if (fetched_at - provider_at).total_seconds() > max_age_seconds:
        raise DojoAiSourceAdapterError("provider evidence is stale")


def _require_market_open(value: datetime) -> None:
    if not compute_market_status(value).is_fx_open:
        raise DojoAiSourceAdapterMarketClosedError(
            "OANDA source acquisition is disabled while the FX market is closed"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: object, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DojoAiSourceAdapterError(f"{field} is invalid") from exc
    else:
        raise DojoAiSourceAdapterError(f"{field} is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoAiSourceAdapterError(f"{field} must be an aware UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise DojoAiSourceAdapterError(f"{field} must be a finite positive number")
    try:
        number = float(value)
    except ValueError as exc:
        raise DojoAiSourceAdapterError(
            f"{field} must be a finite positive number"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise DojoAiSourceAdapterError(f"{field} must be a finite positive number")
    return number


def _exact_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DojoAiSourceAdapterError(
            f"{field} must be an exact integer in [{minimum},{maximum}]"
        )
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


__all__ = [
    "DojoAiSourceAdapterError",
    "DojoAiSourceAdapterMarketClosedError",
    "OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID",
    "OANDA_EXECUTABLE_QUOTE_ADAPTER_ID",
    "OANDA_READ_ONLY_PROVIDER_KIND",
    "SOURCE_ADAPTER_CONFIG_CONTRACT",
    "SourceAdapterAcquisition",
    "acquire_oanda_completed_bid_ask_candles",
    "acquire_oanda_executable_quote",
    "canonical_source_adapter_config_bytes",
    "seal_source_adapter_config",
    "source_adapter_capture_binding",
    "source_adapter_config_sha256",
    "source_adapter_module_sha256",
]
