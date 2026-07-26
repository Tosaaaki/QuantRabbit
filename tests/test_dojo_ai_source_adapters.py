from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.dojo_ai_source_adapters import (
    OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
    OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
    OANDA_READ_ONLY_PROVIDER_KIND,
    SOURCE_ADAPTER_CONFIG_CONTRACT,
    DojoAiSourceAdapterError,
    DojoAiSourceAdapterMarketClosedError,
    acquire_oanda_completed_bid_ask_candles,
    acquire_oanda_executable_quote,
    canonical_source_adapter_config_bytes,
    seal_source_adapter_config,
    source_adapter_capture_binding,
)
from quant_rabbit.models import Quote


def _utc(day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, minute, second, tzinfo=timezone.utc)


def _quote_config(**overrides: object) -> dict[str, object]:
    return seal_source_adapter_config(_quote_config_body(**overrides))


def _quote_config_body(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
        "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
        "pair": "USD_JPY",
        "max_age_seconds": 120,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    value.update(overrides)
    return value


def _candle_config(**overrides: object) -> dict[str, object]:
    return seal_source_adapter_config(_candle_config_body(**overrides))


def _candle_config_body(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
        "adapter_id": OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
        "pair": "USD_JPY",
        "max_age_seconds": 120,
        "granularity": "M1",
        "count": 2,
        "price_component": "BA",
        "smooth": False,
        "complete_only": True,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    value.update(overrides)
    return value


def _candle(
    timestamp: str,
    *,
    complete: bool = True,
) -> dict[str, object]:
    return {
        "time": timestamp,
        "volume": 10,
        "complete": complete,
        "bid": {
            "o": "163.100",
            "h": "163.120",
            "l": "163.090",
            "c": "163.110",
        },
        "ask": {
            "o": "163.110",
            "h": "163.130",
            "l": "163.100",
            "c": "163.120",
        },
    }


class _QuoteClient:
    def __init__(self, quotes: dict[str, Quote]) -> None:
        self.result = quotes
        self.calls: list[tuple[str, ...]] = []

    def quotes(self, pairs: tuple[str, ...]) -> dict[str, Quote]:
        self.calls.append(pairs)
        return self.result


class _CandleClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_json(self, path: str, query: dict[str, str]) -> object:
        self.calls.append((path, query))
        return self.payload


class DojoAiSourceAdaptersTest(unittest.TestCase):
    def test_quote_is_canonical_current_and_watermarked(self) -> None:
        client = _QuoteClient(
            {
                "USD_JPY": Quote(
                    pair="USD_JPY",
                    bid=163.12,
                    ask=163.13,
                    timestamp_utc=_utc(23, 12, 0, 20),
                )
            }
        )
        with (
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                side_effect=(_utc(23, 12, 0, 15), _utc(23, 12, 0, 30)),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                return_value=client,
            ) as factory,
        ):
            result = acquire_oanda_executable_quote(_quote_config())
        factory.assert_called_once_with()
        self.assertEqual(client.calls, [("USD_JPY",)])
        self.assertTrue(result.raw_bytes.endswith(b"\n"))
        self.assertEqual(result.raw_bytes.count(b"\n"), 1)
        payload = json.loads(result.raw_bytes)
        self.assertEqual(
            payload,
            {
                "pair": "USD_JPY",
                "bid": 163.12,
                "ask": 163.13,
                "timestamp_utc": "2026-07-23T12:00:20Z",
                "max_age_seconds": 120,
            },
        )
        self.assertEqual(
            result.source_watermark_sha256,
            hashlib.sha256(result.raw_bytes).hexdigest(),
        )
        self.assertEqual(result.provider_timestamp_utc, payload["timestamp_utc"])
        self.assertEqual(
            result.raw_bytes,
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n",
        )

    def test_completed_bid_ask_candles_use_one_fixed_read_request(self) -> None:
        payload = {
            "instrument": "USD_JPY",
            "granularity": "M1",
            "candles": [
                _candle("2026-07-23T11:59:00Z"),
                _candle("2026-07-23T12:00:00Z"),
            ],
        }
        client = _CandleClient(payload)
        with (
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                side_effect=(_utc(23, 12, 1, 10), _utc(23, 12, 1, 20)),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                return_value=client,
            ),
        ):
            result = acquire_oanda_completed_bid_ask_candles(_candle_config())
        self.assertEqual(
            client.calls,
            [
                (
                    "/v3/instruments/USD_JPY/candles",
                    {
                        "granularity": "M1",
                        "count": "2",
                        "price": "BA",
                        "smooth": "false",
                    },
                )
            ],
        )
        captured = json.loads(result.raw_bytes)
        self.assertIsInstance(captured, list)
        self.assertEqual(len(captured), 2)
        expected_keys = {
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
        self.assertEqual(set(captured[0]), expected_keys)
        self.assertEqual(set(captured[1]), expected_keys)
        self.assertEqual(captured[0]["pair"], "USD_JPY")
        self.assertEqual(captured[0]["granularity"], "M1")
        self.assertEqual(captured[0]["bid_o"], 163.1)
        self.assertEqual(captured[1]["ask_c"], 163.12)
        self.assertEqual(captured[1]["max_age_seconds"], 120)
        self.assertNotIn("volume", captured[0])
        self.assertEqual(
            result.provider_timestamp_utc,
            "2026-07-23T12:01:00Z",
        )
        self.assertEqual(
            result.source_watermark_sha256,
            hashlib.sha256(result.raw_bytes).hexdigest(),
        )
        self.assertNotIn("mid", json.dumps(captured).lower())
        self.assertNotIn("provider_kind", result.raw_bytes.decode())
        self.assertNotIn("adapter_id", result.raw_bytes.decode())
        self.assertNotIn("config_sha256", result.raw_bytes.decode())
        self.assertNotIn("fetched_at_utc", result.raw_bytes.decode())

    def test_weekend_gate_runs_before_client_or_network(self) -> None:
        with (
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                return_value=_utc(25, 12, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as factory,
        ):
            with self.assertRaises(DojoAiSourceAdapterMarketClosedError):
                acquire_oanda_executable_quote(_quote_config())
            with self.assertRaises(DojoAiSourceAdapterMarketClosedError):
                acquire_oanda_completed_bid_ask_candles(_candle_config())
        factory.assert_not_called()

    def test_future_stale_and_crossed_quote_fail_closed(self) -> None:
        cases = (
            (
                "future",
                Quote("USD_JPY", 163.12, 163.13, _utc(23, 12, 0, 31)),
            ),
            (
                "stale",
                Quote("USD_JPY", 163.12, 163.13, _utc(23, 11, 58, 0)),
            ),
            (
                "crossed",
                Quote("USD_JPY", 163.14, 163.13, _utc(23, 12, 0, 20)),
            ),
        )
        for label, quote in cases:
            with self.subTest(label=label):
                with (
                    patch(
                        "quant_rabbit.dojo_ai_source_adapters._utc_now",
                        side_effect=(
                            _utc(23, 12, 0, 15),
                            _utc(23, 12, 0, 30),
                        ),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                        return_value=_QuoteClient({"USD_JPY": quote}),
                    ),
                ):
                    with self.assertRaises(DojoAiSourceAdapterError):
                        acquire_oanda_executable_quote(_quote_config())

    def test_incomplete_future_and_nonexact_candles_fail_closed(self) -> None:
        cases = (
            (
                "incomplete",
                {
                    "instrument": "USD_JPY",
                    "granularity": "M1",
                    "candles": [
                        _candle("2026-07-23T11:59:00Z"),
                        _candle("2026-07-23T12:00:00Z", complete=False),
                    ],
                },
            ),
            (
                "future",
                {
                    "instrument": "USD_JPY",
                    "granularity": "M1",
                    "candles": [
                        _candle("2026-07-23T12:00:00Z"),
                        _candle("2026-07-23T12:01:00Z"),
                    ],
                },
            ),
            (
                "short",
                {
                    "instrument": "USD_JPY",
                    "granularity": "M1",
                    "candles": [_candle("2026-07-23T12:00:00Z")],
                },
            ),
        )
        for label, payload in cases:
            with self.subTest(label=label):
                with (
                    patch(
                        "quant_rabbit.dojo_ai_source_adapters._utc_now",
                        side_effect=(
                            _utc(23, 12, 1, 10),
                            _utc(23, 12, 1, 20),
                        ),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                        return_value=_CandleClient(payload),
                    ),
                ):
                    with self.assertRaises(DojoAiSourceAdapterError):
                        acquire_oanda_completed_bid_ask_candles(_candle_config())

    def test_config_schema_digest_pair_and_mode_are_allowlisted(self) -> None:
        for label, config in (
            ("pair", _quote_config_body(pair="XAU_USD")),
            ("unknown", _quote_config_body(adapter_id="unknown-adapter")),
            ("granularity", _candle_config_body(granularity="D")),
            ("price", _candle_config_body(price_component="M")),
            ("count", _candle_config_body(count=5_001)),
            ("quote age", _quote_config_body(max_age_seconds=181)),
            ("candle age", _candle_config_body(max_age_seconds=86_401)),
        ):
            with self.subTest(label=label):
                with self.assertRaises(DojoAiSourceAdapterError):
                    seal_source_adapter_config(config)

        sealed = _quote_config()
        sealed["config_sha256"] = "0" * 64
        with (
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                return_value=_utc(23, 12, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as factory,
        ):
            with self.assertRaisesRegex(DojoAiSourceAdapterError, "digest"):
                acquire_oanda_executable_quote(sealed)
        factory.assert_not_called()
        extra = {**_quote_config(), "base_url": "https://example.invalid"}
        with self.assertRaisesRegex(DojoAiSourceAdapterError, "schema"):
            canonical_source_adapter_config_bytes(extra)

    def test_capture_binding_binds_module_and_exact_config_bytes(self) -> None:
        config = _quote_config()
        binding = source_adapter_capture_binding(config)
        self.assertEqual(
            binding,
            {
                "source_role": "quote",
                "provider_kind": OANDA_READ_ONLY_PROVIDER_KIND,
                "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
                "adapter_module": "quant_rabbit.dojo_ai_source_adapters",
                "adapter_callable": "acquire_oanda_executable_quote",
                "adapter_executable_sha256": hashlib.sha256(
                    (
                        Path(__file__).resolve().parents[1]
                        / "src/quant_rabbit/dojo_ai_source_adapters.py"
                    ).read_bytes()
                ).hexdigest(),
                "adapter_config_sha256": hashlib.sha256(
                    canonical_source_adapter_config_bytes(config)
                ).hexdigest(),
            },
        )
        candle = source_adapter_capture_binding(_candle_config())
        self.assertEqual(candle["source_role"], "candles")
        self.assertEqual(
            candle["adapter_callable"],
            "acquire_oanda_completed_bid_ask_candles",
        )

    def test_module_imports_only_the_read_only_oanda_client(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "src/quant_rabbit/dojo_ai_source_adapters.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "from quant_rabbit.broker.oanda import OandaReadOnlyClient",
            source,
        )
        self.assertNotIn("OandaExecutionClient", source)
        self.assertNotIn(".post_json(", source)
        self.assertNotIn(".close_trade(", source)
        self.assertNotIn(".create_order(", source)


if __name__ == "__main__":
    unittest.main()
