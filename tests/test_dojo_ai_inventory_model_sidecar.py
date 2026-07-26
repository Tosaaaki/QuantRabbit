from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from quant_rabbit.dojo_ai_inventory_model_sidecar import (
    DEDICATED_MODEL_SIDECAR_CONFIG_ROOT,
    DOJO_AI_MODEL_SIDECAR_CONFIG_CONTRACT,
    AiInventoryModelSidecarConfigError,
    AiInventoryModelSidecarMarketClosedError,
    AiInventoryModelSidecarModelError,
    load_production_adapter_manifest,
    model_sidecar_config_sha256,
    run_model_sidecar,
)
from quant_rabbit.dojo_ai_inventory_producer import (
    AiInventoryProducerModelError,
    load_sealed_command_model_adapter,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 7, day, hour, tzinfo=timezone.utc)


class DojoAiInventoryModelSidecarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo = base / "repo"
        self.repo.mkdir()
        self.config_root = self.repo / DEDICATED_MODEL_SIDECAR_CONFIG_ROOT
        self.config_root.mkdir(parents=True)
        self.marker = base / "model-called"
        self.model_output = base / "model-signed-envelope.json"

        self.private_key = Ed25519PrivateKey.generate()
        private_raw = self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        private_key_base64 = base64.b64encode(private_raw).decode("ascii")
        public_raw = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.public_key_base64 = base64.b64encode(public_raw).decode("ascii")

        self.model_executable = base / "fake-model"
        response = {
            "action": "HOLD",
            "reason_code": "THESIS_ALIVE",
            "reason": "Authenticated point-in-time evidence remains intact.",
            "virtual_units": None,
            "confidence": 0.7,
        }
        source = (
            f"#!{Path(os.sys.executable).resolve()}\n"
            "import base64,hashlib,json,pathlib,sys\n"
            "from cryptography.hazmat.primitives.asymmetric.ed25519 "
            "import Ed25519PrivateKey\n"
            "raw=sys.stdin.buffer.read()\n"
            f"pathlib.Path({str(self.marker)!r}).write_text('called')\n"
            f"response=json.loads({json.dumps(_canonical(response).decode())})\n"
            "request=json.loads(raw)\n"
            "body={'contract':'QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1',"
            "'adapter_id':'signed-sidecar-v1',"
            "'model_id':request['model_id'],"
            "'request_sha256':hashlib.sha256(raw).hexdigest(),"
            "'response':response,"
            "'signature_key_id':'ephemeral-test-key-v1'}\n"
            "payload=json.dumps(body,sort_keys=True,separators=(',',':')).encode()\n"
            f"key=Ed25519PrivateKey.from_private_bytes(base64.b64decode("
            f"{private_key_base64!r}))\n"
            "envelope={**body,'signature_base64':"
            "base64.b64encode(key.sign(payload)).decode()}\n"
            "output=json.dumps(envelope,sort_keys=True,separators=(',',':')).encode()\n"
            f"pathlib.Path({str(self.model_output)!r}).write_bytes(output)\n"
            "sys.stdout.buffer.write(output)\n"
        )
        self.model_executable.write_text(source, encoding="utf-8")
        self.model_executable.chmod(0o700)

        self.sidecar_executable = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run-dojo-ai-inventory-model-sidecar.py"
        )
        self.assertTrue(os.access(self.sidecar_executable, os.X_OK))
        self.git_head = "a" * 40
        self.git_branch = "codex/test-sidecar"
        self.experiment_id = "paper-ai-inventory-experiment-v1"
        self.room_id = "paper-ai-inventory-room-01"
        self.candidate_id = "b" * 64
        self.producer_id = "producer-v1"
        self.future_window = {
            "start_utc": "2026-07-23T00:00:00Z",
            "end_utc": "2026-07-24T00:00:00Z",
        }
        self.config = self._config()
        self.config_path = self._write_config(self.config)
        self.preflight = self._preflight()

    def _identity(self, path: Path) -> dict[str, object]:
        item = path.lstat()
        return {
            "sha": _sha(path),
            "device": item.st_dev,
            "inode": item.st_ino,
            "uid": item.st_uid,
            "gid": item.st_gid,
        }

    def _config(self, **overrides: object) -> dict[str, object]:
        model = self._identity(self.model_executable)
        sidecar = self._identity(self.sidecar_executable)
        value: dict[str, object] = {
            "contract": DOJO_AI_MODEL_SIDECAR_CONFIG_CONTRACT,
            "adapter_id": "signed-sidecar-v1",
            "model_id": "fake-point-in-time-model-v1",
            "producer_id": self.producer_id,
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "room_id": self.room_id,
            "future_window": self.future_window,
            "model_executable_path": str(self.model_executable),
            "model_executable_sha256": model["sha"],
            "model_executable_device": model["device"],
            "model_executable_inode": model["inode"],
            "model_executor_uid": model["uid"],
            "model_executor_gid": model["gid"],
            "model_argv": [str(self.model_executable)],
            "model_timeout_seconds": 10,
            "sidecar_executable_path": str(self.sidecar_executable),
            "sidecar_executable_sha256": sidecar["sha"],
            "sidecar_executable_device": sidecar["device"],
            "sidecar_executable_inode": sidecar["inode"],
            "sidecar_executor_uid": sidecar["uid"],
            "sidecar_executor_gid": sidecar["gid"],
            "sidecar_timeout_seconds": 20,
            "signature_key_id": "ephemeral-test-key-v1",
            "ed25519_public_key_base64": self.public_key_base64,
            "git_head": self.git_head,
            "git_branch": self.git_branch,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        value.update(overrides)
        return value

    def _preflight(self) -> dict[str, object]:
        body: dict[str, object] = {
            "contract": "QR_DOJO_AI_INVENTORY_LAUNCH_PREFLIGHT_V1",
            "adapter_id": "signed-sidecar-v1",
            "model_id": "fake-point-in-time-model-v1",
            "config_sha256": self.config_path.stem,
            "producer_id": self.producer_id,
            "candidate_id": self.candidate_id,
            "source_capture_manifest_sha256": "0" * 64,
            "spec_sha256": "1" * 64,
            "policy_sha256": "2" * 64,
            "experiment_id": self.experiment_id,
            "room_id": self.room_id,
            "paper_eligible_event_sha256": "3" * 64,
            "candidate_lifecycle_ledger_tip_sha256": "3" * 64,
            "append_claim_sha256": "4" * 64,
            "job_manifest_sha256": "5" * 64,
            "job_owner_sha256": "6" * 64,
            "proof_artifact_sha256": "7" * 64,
            "proof_artifact_bytes_sha256": "8" * 64,
            "proof_manifest_sha256": "9" * 64,
            "replay_worker_receipt_sha256": "a" * 64,
            "source_manifest_sha256s": {
                "TRAIN": "c" * 64,
                "VAL": "d" * 64,
                "S5": "e" * 64,
            },
            "future_registry_sha256": "f" * 64,
            "future_window": self.future_window,
            "git_head": self.git_head,
            "git_head_sha256": hashlib.sha256(self.git_head.encode()).hexdigest(),
            "issued_at_utc": "2026-07-22T23:00:00Z",
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "paper_room_launched": False,
        }
        return {
            **body,
            "launch_preflight_token_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }

    def _write_config(self, value: dict[str, object]) -> Path:
        digest = model_sidecar_config_sha256(value)
        sealed = {**value, "config_sha256": digest}
        path = self.config_root / f"{digest}.json"
        path.write_bytes(_canonical(sealed) + b"\n")
        return path

    def _install_config(self, value: dict[str, object]) -> None:
        self.config = value
        self.config_path = self._write_config(value)
        self.preflight = self._preflight()

    def _request(self) -> bytes:
        return _canonical(
            {
                "contract": "QR_DOJO_AI_INVENTORY_PROPOSAL_REQUEST_V2",
                "producer_id": "producer-v1",
                "model_id": "fake-point-in-time-model-v1",
                "purpose": "PAPER_AI_INVENTORY_PROPOSAL_ONLY",
                "evidence_packet": {"packet_sha256": "c" * 64},
                "source_watermarks": {},
                "source_watermarks_sha256": "d" * 64,
                "required_response": {},
                "safety": {
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                    "proposal_is_not_an_action": True,
                    "arbitrary_prose_has_no_authority": True,
                },
            }
        )

    def _run(self, *, now: datetime | None = None) -> bytes:
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._trusted_repository_root",
                return_value=self.repo,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._read_git_identity",
                return_value=(self.git_head, self.git_branch),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._utc_now",
                return_value=now or _utc(23),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
        ):
            return run_model_sidecar(
                self.config_path.stem,
                self._request(),
            )

    def test_runs_fixed_model_and_returns_verifiable_signed_envelope(self) -> None:
        raw = self._run()
        envelope = json.loads(raw)
        signature = base64.b64decode(envelope.pop("signature_base64"), validate=True)
        Ed25519PublicKey.from_public_bytes(
            base64.b64decode(self.public_key_base64, validate=True)
        ).verify(signature, _canonical(envelope))
        self.assertEqual(envelope["adapter_id"], "signed-sidecar-v1")
        self.assertEqual(
            envelope["request_sha256"],
            hashlib.sha256(self._request()).hexdigest(),
        )
        self.assertEqual(envelope["response"]["action"], "HOLD")
        self.assertTrue(self.marker.exists())
        self.assertEqual(raw, self.model_output.read_bytes())
        self.assertNotIn("PRIVATE KEY", raw.decode())

    def test_weekend_stops_before_nested_model(self) -> None:
        with self.assertRaises(AiInventoryModelSidecarMarketClosedError):
            self._run(now=_utc(25))
        self.assertFalse(self.marker.exists())

    def test_executable_identity_or_content_change_fails_closed(self) -> None:
        self.model_executable.write_text(
            self.model_executable.read_text(encoding="utf-8") + "# changed\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            AiInventoryModelSidecarConfigError,
            "identity or digest",
        ):
            self._run()
        self.assertFalse(self.marker.exists())

    def test_symlinked_nested_model_executable_fails_closed(self) -> None:
        target = Path(self.temp.name) / "real-fake-model"
        self.model_executable.rename(target)
        self.model_executable.symlink_to(target)
        config = self._config()
        self._install_config(config)
        with self.assertRaisesRegex(
            AiInventoryModelSidecarConfigError,
            "real regular file",
        ):
            self._run()
        self.assertFalse(self.marker.exists())

    def test_config_tamper_and_symlink_fail_closed(self) -> None:
        original = self.config_path.read_bytes()
        self.config_path.write_bytes(
            original.replace(b"signed-sidecar-v1", b"signed-sidecar-v2")
        )
        with self.assertRaisesRegex(
            AiInventoryModelSidecarConfigError,
            "digest mismatch",
        ):
            self._run()

        self.config_path.unlink()
        target = Path(self.temp.name) / "config-target.json"
        target.write_bytes(original)
        self.config_path.symlink_to(target)
        with self.assertRaises(AiInventoryModelSidecarConfigError):
            self._run()

    def test_sidecar_config_contains_only_the_model_public_key(self) -> None:
        sealed = json.loads(self.config_path.read_bytes())
        self.assertTrue(all("private" not in key for key in sealed))
        self.assertEqual(
            sealed["ed25519_public_key_base64"],
            self.public_key_base64,
        )

    def test_noncanonical_or_unknown_model_response_fails_closed(self) -> None:
        source = (
            f"#!{Path(os.sys.executable).resolve()}\n"
            "import sys\n"
            "sys.stdin.buffer.read()\n"
            'sys.stdout.write(\'{"action":"HOLD","debug":true}\')\n'
        )
        self.model_executable.write_text(source, encoding="utf-8")
        self.model_executable.chmod(0o700)
        self._install_config(self._config())
        with self.assertRaisesRegex(
            AiInventoryModelSidecarModelError,
            "schema",
        ):
            self._run()

    def test_signed_response_tamper_is_detectable(self) -> None:
        envelope = json.loads(self._run())
        signature = base64.b64decode(
            envelope.pop("signature_base64"),
            validate=True,
        )
        envelope["response"]["action"] = "CLOSE_VIRTUAL"
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(self.public_key_base64, validate=True)
        )
        with self.assertRaises(InvalidSignature):
            public_key.verify(signature, _canonical(envelope))

    def test_authenticated_malicious_model_without_key_cannot_forge(self) -> None:
        source = (
            f"#!{Path(os.sys.executable).resolve()}\n"
            "import base64,hashlib,json,sys\n"
            "raw=sys.stdin.buffer.read()\n"
            "request=json.loads(raw)\n"
            "response={'action':'CLOSE_VIRTUAL',"
            "'reason_code':'FORGED_CLOSE',"
            "'reason':'Authenticated executable lacks the model signing key.',"
            "'virtual_units':1000.0,'confidence':1.0}\n"
            "envelope={'contract':"
            "'QR_DOJO_AI_INVENTORY_SIGNED_MODEL_RESPONSE_V1',"
            "'adapter_id':'signed-sidecar-v1',"
            "'model_id':request['model_id'],"
            "'request_sha256':hashlib.sha256(raw).hexdigest(),"
            "'response':response,"
            "'signature_key_id':'ephemeral-test-key-v1',"
            "'signature_base64':base64.b64encode(bytes(64)).decode()}\n"
            "sys.stdout.write(json.dumps("
            "envelope,sort_keys=True,separators=(',',':')))\n"
        )
        self.model_executable.write_text(source, encoding="utf-8")
        self.model_executable.chmod(0o700)
        self._install_config(self._config())

        with self.assertRaisesRegex(
            AiInventoryModelSidecarModelError,
            "signature is invalid",
        ):
            self._run()

    def test_production_loader_requires_matching_adapter_git_and_preflight(
        self,
    ) -> None:
        patches = (
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._trusted_repository_root",
                return_value=self.repo,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._read_git_identity",
                return_value=(self.git_head, self.git_branch),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._utc_now",
                return_value=_utc(23),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            registration = load_production_adapter_manifest(
                "signed-sidecar-v1",
                self.config_path.stem,
                experiment_id=self.experiment_id,
                room_id=self.room_id,
            )
            manifest = registration["command_manifest"]
            binding = registration["lifecycle_binding"]
            self.assertNotIn("private", _canonical(registration).decode())
            self.assertEqual(
                manifest["argv"],
                [
                    str(self.sidecar_executable),
                    "--config-sha256",
                    self.config_path.stem,
                ],
            )
            self.assertEqual(binding["adapter_id"], "signed-sidecar-v1")
            self.assertEqual(binding["model_id"], "fake-point-in-time-model-v1")
            self.assertEqual(binding["producer_id"], self.producer_id)
            self.assertEqual(binding["config_sha256"], self.config_path.stem)
            with self.assertRaises(AiInventoryModelSidecarConfigError):
                load_production_adapter_manifest(
                    "unknown-adapter",
                    self.config_path.stem,
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                )

    def test_canonical_preflight_candidate_window_and_git_are_bound(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        adapter = json.loads(_canonical(self.preflight))
        adapter["adapter_id"] = "other-sidecar-v1"
        cases.append(("adapter", self._reseal_preflight(adapter)))

        model = json.loads(_canonical(self.preflight))
        model["model_id"] = "other-model-v1"
        cases.append(("model", self._reseal_preflight(model)))

        config = json.loads(_canonical(self.preflight))
        config["config_sha256"] = "0" * 64
        cases.append(("config", self._reseal_preflight(config)))

        producer = json.loads(_canonical(self.preflight))
        producer["producer_id"] = "other-producer-v1"
        cases.append(("producer", self._reseal_preflight(producer)))

        candidate = json.loads(_canonical(self.preflight))
        candidate["candidate_id"] = "0" * 64
        cases.append(("candidate", self._reseal_preflight(candidate)))

        git = json.loads(_canonical(self.preflight))
        git["git_head"] = "0" * 40
        cases.append(("Git", self._reseal_preflight(git)))

        expired = json.loads(_canonical(self.preflight))
        expired["future_window"] = {
            "start_utc": "2026-07-21T00:00:00Z",
            "end_utc": "2026-07-22T00:00:00Z",
        }
        expired["issued_at_utc"] = "2026-07-20T23:00:00Z"
        cases.append(("future window", self._reseal_preflight(expired)))

        for label, token in cases:
            with self.subTest(label=label):
                with (
                    patch(
                        "quant_rabbit.dojo_ai_inventory_model_sidecar."
                        "_trusted_repository_root",
                        return_value=self.repo,
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_model_sidecar."
                        "_read_git_identity",
                        return_value=(self.git_head, self.git_branch),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_model_sidecar._utc_now",
                        return_value=_utc(23),
                    ),
                    patch(
                        "quant_rabbit.dojo_ai_inventory_model_sidecar."
                        "verify_paper_ai_inventory_launch_preflight",
                        return_value=token,
                    ),
                ):
                    with self.assertRaises(AiInventoryModelSidecarConfigError):
                        load_production_adapter_manifest(
                            "signed-sidecar-v1",
                            self.config_path.stem,
                            experiment_id=self.experiment_id,
                            room_id=self.room_id,
                        )

    def test_producer_sealed_loader_exposes_only_opaque_adapter(self) -> None:
        with (
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._trusted_repository_root",
                return_value=self.repo,
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._read_git_identity",
                return_value=(self.git_head, self.git_branch),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar._utc_now",
                return_value=_utc(23),
            ),
            patch(
                "quant_rabbit.dojo_ai_inventory_model_sidecar."
                "verify_paper_ai_inventory_launch_preflight",
                return_value=self.preflight,
            ),
            patch.dict(
                "quant_rabbit.dojo_ai_inventory_producer." "_LOADED_COMMAND_ADAPTERS",
                {},
                clear=True,
            ),
            patch.dict(
                "quant_rabbit.dojo_ai_inventory_producer."
                "_LOADED_COMMAND_ADAPTER_BINDINGS",
                {},
                clear=True,
            ),
        ):
            adapter = load_sealed_command_model_adapter(
                "signed-sidecar-v1",
                self.config_path.stem,
                experiment_id=self.experiment_id,
                room_id=self.room_id,
            )
            self.assertEqual(adapter.adapter_id, "signed-sidecar-v1")
            with self.assertRaises(AiInventoryProducerModelError):
                load_sealed_command_model_adapter(
                    "unknown-adapter",
                    self.config_path.stem,
                    experiment_id=self.experiment_id,
                    room_id=self.room_id,
                )

    @staticmethod
    def _reseal_preflight(value: dict[str, object]) -> dict[str, object]:
        body = {
            key: item
            for key, item in value.items()
            if key != "launch_preflight_token_sha256"
        }
        return {
            **body,
            "launch_preflight_token_sha256": hashlib.sha256(
                _canonical(body)
            ).hexdigest(),
        }


if __name__ == "__main__":
    unittest.main()
