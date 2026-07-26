from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_rabbit.dojo_ai_source_adapters import (
    OANDA_COMPLETED_BID_ASK_CANDLES_ADAPTER_ID,
    OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
    SOURCE_ADAPTER_CONFIG_CONTRACT,
    canonical_source_adapter_config_bytes,
    seal_source_adapter_config,
    source_adapter_capture_binding,
)
from quant_rabbit.dojo_ai_source_capture import (
    CANONICAL_SOURCE_ROOT,
    CAPTURE_PRIVATE_KEY_ENV,
    CAPTURE_ROOT,
    SOURCE_CAPTURE_MANIFEST_CONTRACT,
    AiSourceCaptureError,
    AiSourceCaptureMarketClosedError,
    ReadOnlySourceAcquisition,
    capture_registered_ai_source,
    capture_test_only_ai_source,
    _normalize_acquisition,
    source_capture_manifest_sha256,
    verify_ai_source_capture_receipt,
    verify_test_only_ai_source_capture_receipt,
)
from quant_rabbit.dojo_ai_evidence_packet import (
    build_trusted_ai_inventory_evidence_packet,
)
from quant_rabbit.models import Quote
from tests.test_dojo_ai_evidence_packet import (
    _TEST_GIT_BRANCH,
    _TEST_GIT_HEAD,
    _paper_eligible_preflight,
    _trusted_request,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run-dojo-ai-source-capture.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_dojo_ai_source_capture",
    _SCRIPT,
)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"cannot load script: {_SCRIPT}")
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _utc(day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(
        2026,
        7,
        day,
        hour,
        minute,
        second,
        tzinfo=timezone.utc,
    )


class DojoAiSourceCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_context = tempfile.TemporaryDirectory()
        self.key_context = tempfile.TemporaryDirectory()
        self.repository = Path(self.repository_context.name).resolve()
        self.repository.joinpath(".git").mkdir()
        self.private_key = Ed25519PrivateKey.generate()
        public_raw = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        self.public_key_base64 = base64.b64encode(public_raw).decode("ascii")
        manifest_body = {
            "contract": SOURCE_CAPTURE_MANIFEST_CONTRACT,
            "manifest_id": "paper-ai-source-capture-v1",
            "capture_key_id": "paper-ai-capture-key-v1",
            "ed25519_public_key_base64": self.public_key_base64,
            "allowed_source_roles": ["quote", "state"],
            "allowed_provider_kinds": ["LOCAL_READ_ONLY"],
            "source_adapters": [
                {
                    "source_role": role,
                    "provider_kind": "LOCAL_READ_ONLY",
                    "adapter_id": "TEST_ONLY_CALLBACK",
                    "adapter_module": "tests.test_dojo_ai_source_capture",
                    "adapter_callable": "acquisition",
                    "adapter_executable_sha256": "a" * 64,
                    "adapter_config_sha256": "b" * 64,
                }
                for role in ("quote", "state")
            ],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": source_capture_manifest_sha256(manifest_body),
        }
        manifest_raw = _canonical_bytes(manifest) + b"\n"
        self.manifest_file_sha = hashlib.sha256(manifest_raw).hexdigest()
        manifest_root = self.repository / CAPTURE_ROOT / "manifests"
        manifest_root.mkdir(parents=True)
        manifest_root.joinpath(f"{self.manifest_file_sha}.json").write_bytes(
            manifest_raw
        )
        self.key_path = Path(self.key_context.name).joinpath("capture-key.pem")
        self.key_path.write_bytes(
            self.private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        self.experiment_id = "paper-ai-inventory-v1"
        self.room_id = "paper-ai-inventory-room-01"
        self.candidate_id = "c" * 64
        self.token = {
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "room_id": self.room_id,
            "source_capture_manifest_sha256": self.manifest_file_sha,
            "future_window": {
                "start_utc": "2026-07-23T11:00:00Z",
                "end_utc": "2026-07-23T13:00:00Z",
            },
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }

    def _install_registered_manifest(
        self, config: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, str]]:
        config_raw = canonical_source_adapter_config_bytes(config)
        config_sha = hashlib.sha256(config_raw).hexdigest()
        config_root = self.repository / CAPTURE_ROOT / "adapter_configs"
        config_root.mkdir(parents=True, exist_ok=True)
        config_root.joinpath(f"{config_sha}.json").write_bytes(config_raw)
        binding = source_adapter_capture_binding(config)
        self.assertEqual(binding["adapter_config_sha256"], config_sha)
        manifest_body = {
            "contract": SOURCE_CAPTURE_MANIFEST_CONTRACT,
            "manifest_id": "paper-ai-oanda-source-capture-v1",
            "capture_key_id": "paper-ai-capture-key-v1",
            "ed25519_public_key_base64": self.public_key_base64,
            "allowed_source_roles": [binding["source_role"]],
            "allowed_provider_kinds": [binding["provider_kind"]],
            "source_adapters": [binding],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": source_capture_manifest_sha256(manifest_body),
        }
        raw = _canonical_bytes(manifest) + b"\n"
        file_sha = hashlib.sha256(raw).hexdigest()
        path = self.repository / CAPTURE_ROOT / "manifests" / f"{file_sha}.json"
        path.write_bytes(raw)
        return {
            **self.token,
            "source_capture_manifest_sha256": file_sha,
        }, binding

    def _capture_registered_quote(
        self,
        token: dict[str, object],
        *,
        client: Mock | None = None,
    ) -> tuple[dict[str, object], Mock]:
        quote_client = client or Mock()
        quote_client.quotes.return_value = {
            "USD_JPY": Quote(
                "USD_JPY",
                163.12,
                163.13,
                _utc(23, 11, 59, 30),
            )
        }
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                side_effect=(
                    _utc(23, 11, 59, 10),
                    _utc(23, 11, 59, 40),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                side_effect=(
                    _utc(23, 11, 59, 15),
                    _utc(23, 11, 59, 30),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                return_value=quote_client,
            ),
            patch.dict(
                os.environ,
                {CAPTURE_PRIVATE_KEY_ENV: str(self.key_path)},
            ),
        ):
            receipt = capture_registered_ai_source(
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                source_role="quote",
                cutoff_utc="2026-07-23T12:00:00Z",
            )
        return receipt, quote_client

    def tearDown(self) -> None:
        self.key_context.cleanup()
        self.repository_context.cleanup()

    def _acquisition(
        self,
        *,
        raw: bytes = b'{"ask":163.13,"bid":163.12,"pair":"USD_JPY"}\n',
        provider_timestamp: str = "2026-07-23T11:59:25Z",
        watermark: str = "d" * 64,
    ) -> ReadOnlySourceAcquisition:
        return ReadOnlySourceAcquisition(
            raw_bytes=raw,
            provider_timestamp_utc=provider_timestamp,
            source_watermark_sha256=watermark,
        )

    def _capture(
        self,
        *,
        source_role: str = "quote",
        clocks: tuple[datetime, datetime] = (
            _utc(23, 11, 59, 20),
            _utc(23, 11, 59, 30),
        ),
        acquire: Mock | None = None,
        token: dict[str, object] | None = None,
        key_path: Path | None = None,
    ) -> tuple[dict[str, object], Mock]:
        callback = acquire or Mock(return_value=self._acquisition())
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token or self.token,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                side_effect=clocks,
            ),
            patch.dict(
                os.environ,
                {
                    CAPTURE_PRIVATE_KEY_ENV: str(
                        key_path or self.key_path
                    )
                },
            ),
        ):
            receipt = capture_test_only_ai_source(
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                source_role=source_role,
                cutoff_utc="2026-07-23T12:00:00Z",
                provider_kind="LOCAL_READ_ONLY",
                acquire=callback,
            )
        return receipt, callback

    def _verify(
        self,
        receipt: dict[str, object],
        *,
        source_role: str = "quote",
    ) -> dict[str, object]:
        with patch(
            "quant_rabbit.dojo_ai_source_capture."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=self.token,
        ):
            return verify_test_only_ai_source_capture_receipt(
                self.repository,
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                cutoff_utc="2026-07-23T12:00:00Z",
                source_role=source_role,
                source_sha256=receipt["canonical_source_sha256"],
                receipt_sha256=receipt["receipt_sha256"],
            )

    def test_capture_is_signed_chained_and_mtime_independent(self) -> None:
        first, first_callback = self._capture()
        second, second_callback = self._capture(
            source_role="state",
            clocks=(
                _utc(23, 11, 59, 31),
                _utc(23, 11, 59, 32),
            ),
            acquire=Mock(
                return_value=self._acquisition(
                    raw=b'{"room_id":"paper-ai-inventory-room-01"}\n',
                    provider_timestamp="2026-07-23T11:59:31Z",
                    watermark="e" * 64,
                )
            ),
        )
        self.assertEqual(first_callback.call_count, 1)
        self.assertEqual(second_callback.call_count, 1)
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(
            second["previous_receipt_sha256"], first["receipt_sha256"]
        )
        self.assertEqual(
            self._verify(first)["receipt_sha256"], first["receipt_sha256"]
        )
        self.assertEqual(
            self._verify(second, source_role="state")["receipt_sha256"],
            second["receipt_sha256"],
        )

        future_ns = int(_utc(24, 12, 0).timestamp()) * 1_000_000_000
        source = (
            self.repository
            / CANONICAL_SOURCE_ROOT
            / f"{first['canonical_source_sha256']}.json"
        )
        os.utime(source, ns=(future_ns, future_ns))
        receipt_root = (
            self.repository
            / CAPTURE_ROOT
            / "receipts"
            / self.experiment_id
            / self.room_id
        )
        for path in receipt_root.glob("*.json"):
            os.utime(path, ns=(future_ns, future_ns))
        self.assertEqual(
            self._verify(first)["receipt_sha256"], first["receipt_sha256"]
        )

    def test_weekend_fails_before_fetch_or_capture(self) -> None:
        callback = Mock(return_value=self._acquisition())
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                return_value=_utc(25, 12, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.token,
            ) as preflight,
            patch.dict(
                os.environ,
                {CAPTURE_PRIVATE_KEY_ENV: str(self.key_path)},
            ),
        ):
            with self.assertRaises(AiSourceCaptureMarketClosedError):
                capture_test_only_ai_source(
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                    candidate_id=self.candidate_id,
                    source_role="quote",
                    cutoff_utc="2026-07-25T12:30:00Z",
                    provider_kind="LOCAL_READ_ONLY",
                    acquire=callback,
                )
        callback.assert_not_called()
        preflight.assert_not_called()
        self.assertFalse(
            (self.repository / CAPTURE_ROOT / "receipts").exists()
        )

    def test_missing_manifest_binding_and_wrong_key_fail_before_fetch(self) -> None:
        callback = Mock(return_value=self._acquisition())
        token = dict(self.token)
        token.pop("source_capture_manifest_sha256")
        with self.assertRaisesRegex(
            AiSourceCaptureError, "capture manifest file sha256"
        ):
            self._capture(acquire=callback, token=token)
        callback.assert_not_called()

        wrong_key = Ed25519PrivateKey.generate()
        wrong_path = Path(self.key_context.name).joinpath("wrong-key.pem")
        wrong_path.write_bytes(
            wrong_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        callback.reset_mock()
        with self.assertRaisesRegex(
            AiSourceCaptureError, "does not match bound manifest"
        ):
            self._capture(acquire=callback, key_path=wrong_path)
        callback.assert_not_called()

    def test_tampered_receipt_signature_is_rejected(self) -> None:
        receipt, _ = self._capture()
        receipt_root = (
            self.repository
            / CAPTURE_ROOT
            / "receipts"
            / self.experiment_id
            / self.room_id
        )
        original = next(receipt_root.glob("*.json"))
        tampered = json.loads(original.read_text())
        tampered["provider_timestamp_utc"] = "2026-07-23T11:59:26Z"
        body = {
            key: value
            for key, value in tampered.items()
            if key not in {"receipt_sha256", "signature_base64"}
        }
        tampered_sha = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        tampered["receipt_sha256"] = tampered_sha
        original.unlink()
        receipt_path = receipt_root / f"00000001-{tampered_sha}.json"
        receipt_path.write_bytes(_canonical_bytes(tampered) + b"\n")
        receipt["receipt_sha256"] = tampered_sha
        with self.assertRaisesRegex(AiSourceCaptureError, "signature"):
            self._verify(receipt)

    def test_noncanonical_or_future_provider_result_is_rejected(self) -> None:
        with self.assertRaisesRegex(AiSourceCaptureError, "canonical"):
            self._capture(
                acquire=Mock(
                    return_value=self._acquisition(raw=b'{"pair": "USD_JPY"}\n')
                )
            )
        with self.assertRaisesRegex(
            AiSourceCaptureError, "provider timestamp is after fetch"
        ):
            self._capture(
                acquire=Mock(
                    return_value=self._acquisition(
                        provider_timestamp="2026-07-23T12:00:00Z"
                    )
                )
            )

    def test_production_api_rejects_unregistered_role(self) -> None:
        with self.assertRaisesRegex(AiSourceCaptureError, "no code-owned"):
            capture_registered_ai_source(
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                source_role="state",
                cutoff_utc="2026-07-23T12:00:00Z",
            )

    def test_acquisition_normalization_is_structural_and_exact(self) -> None:
        @dataclass(frozen=True)
        class IndependentAcquisition:
            raw_bytes: bytes
            provider_timestamp_utc: str
            source_watermark_sha256: str

        accepted = _normalize_acquisition(
            IndependentAcquisition(
                raw_bytes=b"{}\n",
                provider_timestamp_utc="2026-07-23T11:59:30Z",
                source_watermark_sha256="a" * 64,
            )
        )
        self.assertIsInstance(accepted, ReadOnlySourceAcquisition)
        with self.assertRaisesRegex(AiSourceCaptureError, "schema"):
            _normalize_acquisition(
                {
                    "raw_bytes": b"{}\n",
                    "provider_timestamp_utc": "2026-07-23T11:59:30Z",
                    "source_watermark_sha256": "a" * 64,
                    "unexpected": True,
                }
            )

    def test_registered_quote_capture_binds_config_code_and_receipt(self) -> None:
        config = seal_source_adapter_config(
            {
                "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
                "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
                "pair": "USD_JPY",
                "max_age_seconds": 120,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        )
        token, binding = self._install_registered_manifest(config)
        receipt, client = self._capture_registered_quote(token)
        self.assertEqual(client.quotes.call_args.args, (("USD_JPY",),))
        for field, expected in binding.items():
            if field != "source_role":
                self.assertEqual(receipt[field], expected)
        with patch(
            "quant_rabbit.dojo_ai_source_capture."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=token,
        ):
            verified = verify_ai_source_capture_receipt(
                self.repository,
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                cutoff_utc="2026-07-23T12:00:00Z",
                source_role="quote",
                source_sha256=receipt["canonical_source_sha256"],
                receipt_sha256=receipt["receipt_sha256"],
            )
        self.assertEqual(verified["receipt_sha256"], receipt["receipt_sha256"])
        source_path = (
            self.repository
            / CANONICAL_SOURCE_ROOT
            / f"{receipt['canonical_source_sha256']}.json"
        )
        self.assertEqual(
            json.loads(source_path.read_bytes()),
            {
                "ask": 163.13,
                "bid": 163.12,
                "max_age_seconds": 120,
                "pair": "USD_JPY",
                "timestamp_utc": "2026-07-23T11:59:30Z",
            },
        )

    def test_registered_config_and_module_tamper_fail_before_network(
        self,
    ) -> None:
        config = seal_source_adapter_config(
            {
                "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
                "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
                "pair": "USD_JPY",
                "max_age_seconds": 120,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        )
        token, binding = self._install_registered_manifest(config)
        config_path = (
            self.repository
            / CAPTURE_ROOT
            / "adapter_configs"
            / f"{binding['adapter_config_sha256']}.json"
        )
        config_path.write_bytes(b"{}\n")
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                return_value=_utc(23, 11, 59, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {CAPTURE_PRIVATE_KEY_ENV: str(self.key_path)},
            ),
        ):
            with self.assertRaisesRegex(
                AiSourceCaptureError, "config digest mismatch"
            ):
                capture_registered_ai_source(
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                    candidate_id=self.candidate_id,
                    source_role="quote",
                    cutoff_utc="2026-07-23T12:00:00Z",
                )
        client_factory.assert_not_called()

        token, _ = self._install_registered_manifest(config)
        fake_module = self.repository / "tampered_adapter.py"
        fake_module.write_bytes(b"# tampered\n")
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                return_value=_utc(23, 11, 59, 10),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture.inspect.getsourcefile",
                return_value=str(fake_module),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
            patch.dict(
                os.environ,
                {CAPTURE_PRIVATE_KEY_ENV: str(self.key_path)},
            ),
        ):
            with self.assertRaisesRegex(
                AiSourceCaptureError, "executable digest mismatch"
            ):
                capture_registered_ai_source(
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                    candidate_id=self.candidate_id,
                    source_role="quote",
                    cutoff_utc="2026-07-23T12:00:00Z",
                )
        client_factory.assert_not_called()

    def test_registered_candle_capture_is_manifest_bound(self) -> None:
        config = seal_source_adapter_config(
            {
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
        )
        token, binding = self._install_registered_manifest(config)
        client = Mock()
        client.get_json.return_value = {
            "instrument": "USD_JPY",
            "granularity": "M1",
            "candles": [
                {
                    "time": "2026-07-23T11:59:00Z",
                    "volume": 10,
                    "complete": True,
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
                },
                {
                    "time": "2026-07-23T12:00:00Z",
                    "volume": 10,
                    "complete": True,
                    "bid": {
                        "o": "163.110",
                        "h": "163.130",
                        "l": "163.100",
                        "c": "163.120",
                    },
                    "ask": {
                        "o": "163.120",
                        "h": "163.140",
                        "l": "163.110",
                        "c": "163.130",
                    },
                },
            ],
        }
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                side_effect=(
                    _utc(23, 12, 1, 10),
                    _utc(23, 12, 1, 40),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters._utc_now",
                side_effect=(
                    _utc(23, 12, 1, 15),
                    _utc(23, 12, 1, 20),
                ),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient",
                return_value=client,
            ),
            patch.dict(
                os.environ,
                {CAPTURE_PRIVATE_KEY_ENV: str(self.key_path)},
            ),
        ):
            receipt = capture_registered_ai_source(
                experiment_id=self.experiment_id,
                room_id=self.room_id,
                candidate_id=self.candidate_id,
                source_role="candles",
                cutoff_utc="2026-07-23T12:02:00Z",
            )
        self.assertEqual(receipt["adapter_id"], binding["adapter_id"])
        self.assertEqual(
            receipt["source_watermark_sha256"],
            receipt["canonical_source_sha256"],
        )
        client.get_json.assert_called_once_with(
            "/v3/instruments/USD_JPY/candles",
            {
                "granularity": "M1",
                "count": "2",
                "price": "BA",
                "smooth": "false",
            },
        )

    def test_registered_quote_receipt_enters_trusted_evidence_packet(
        self,
    ) -> None:
        request = _trusted_request(self.repository)
        config = seal_source_adapter_config(
            {
                "contract": SOURCE_ADAPTER_CONFIG_CONTRACT,
                "adapter_id": OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
                "pair": "USD_JPY",
                "max_age_seconds": 120,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        )
        capture_token, _ = self._install_registered_manifest(config)
        token = _paper_eligible_preflight(request)
        token["source_capture_manifest_sha256"] = capture_token[
            "source_capture_manifest_sha256"
        ]
        token_body = {
            key: value
            for key, value in token.items()
            if key != "launch_preflight_token_sha256"
        }
        token["launch_preflight_token_sha256"] = hashlib.sha256(
            _canonical_bytes(token_body)
        ).hexdigest()
        receipt, _ = self._capture_registered_quote(token)
        self.assertEqual(
            request["source_files"]["quote"],
            f"{receipt['canonical_source_sha256']}.json",
        )
        request["source_receipts"]["quote"] = receipt["receipt_sha256"]

        def verify_receipt(
            repository_root: Path, **kwargs: object
        ) -> dict[str, object]:
            if kwargs["source_role"] == "quote":
                return verify_ai_source_capture_receipt(
                    repository_root,
                    **kwargs,
                )
            return {
                "contract": "QR_DOJO_AI_SOURCE_CAPTURE_RECEIPT_V1",
                "receipt_sha256": kwargs["receipt_sha256"],
            }

        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_evidence_packet._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_evidence_packet._read_git_identity",
                return_value=(_TEST_GIT_HEAD, _TEST_GIT_BRANCH),
            ),
            patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_utc(23, 12, 0, 30),
            ),
            patch(
                "quant_rabbit.dojo_ai_evidence_packet."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=token,
            ),
            patch(
                "quant_rabbit.dojo_ai_evidence_packet."
                "verify_ai_source_capture_receipt",
                side_effect=verify_receipt,
            ),
        ):
            packet = build_trusted_ai_inventory_evidence_packet(request)
        self.assertEqual(packet["quote"]["bid"], 163.12)
        self.assertEqual(packet["quote"]["ask"], 163.13)
        self.assertEqual(
            packet["quote"]["source_sha256"],
            receipt["canonical_source_sha256"],
        )

    def test_registered_capture_weekend_stops_before_preflight_or_network(
        self,
    ) -> None:
        with (
            patch(
                "quant_rabbit.dojo_ai_source_capture._trusted_repository_root",
                return_value=self.repository,
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture._utc_now",
                return_value=_utc(25, 12, 0),
            ),
            patch(
                "quant_rabbit.dojo_ai_source_capture."
                "verify_paper_ai_inventory_launch_preflight"
            ) as preflight,
            patch(
                "quant_rabbit.dojo_ai_source_adapters.OandaReadOnlyClient"
            ) as client_factory,
        ):
            with self.assertRaises(AiSourceCaptureMarketClosedError):
                capture_registered_ai_source(
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                    candidate_id=self.candidate_id,
                    source_role="quote",
                    cutoff_utc="2026-07-25T12:30:00Z",
                )
        preflight.assert_not_called()
        client_factory.assert_not_called()

    def test_production_verifier_rejects_test_only_receipt(self) -> None:
        receipt, _ = self._capture()
        with patch(
            "quant_rabbit.dojo_ai_source_capture."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=self.token,
        ):
            with self.assertRaisesRegex(
                AiSourceCaptureError, "not production evidence"
            ):
                verify_ai_source_capture_receipt(
                    self.repository,
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                    candidate_id=self.candidate_id,
                    cutoff_utc="2026-07-23T12:00:00Z",
                    source_role="quote",
                    source_sha256=receipt["canonical_source_sha256"],
                    receipt_sha256=receipt["receipt_sha256"],
                )

    def test_cli_exposes_only_registered_adapter_capture(self) -> None:
        output = io.StringIO()

        def fake_capture(**kwargs: object) -> dict[str, object]:
            self.assertEqual(kwargs["experiment_id"], self.experiment_id)
            self.assertEqual(kwargs["room_id"], self.room_id)
            self.assertEqual(kwargs["candidate_id"], self.candidate_id)
            self.assertNotIn("acquire", kwargs)
            self.assertNotIn("provider_timestamp_utc", kwargs)
            return {"receipt_sha256": "e" * 64}

        with patch.object(
            _SCRIPT_MODULE,
            "capture_registered_ai_source",
            side_effect=fake_capture,
        ):
            result = _SCRIPT_MODULE.main(
                [
                    "--experiment-id",
                    self.experiment_id,
                    "--room-id",
                    self.room_id,
                    "--candidate-id",
                    self.candidate_id,
                    "--source-role",
                    "quote",
                    "--cutoff-utc",
                    "2026-07-23T12:00:00Z",
                ],
                stdout=output,
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            {"receipt_sha256": "e" * 64},
        )


if __name__ == "__main__":
    unittest.main()
