from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.dojo_ai_evidence_packet import (
    DEDICATED_CANONICAL_SOURCE_ROOT,
    DEDICATED_EVIDENCE_ROOT,
    DOJO_AI_EVIDENCE_PACKET_CONTRACT,
    EvidencePacketIntegrityError,
    EvidencePacketMarketClosedError,
    LOW_LEVEL_BUILDER_LAUNCH_SAFE,
    build_ai_inventory_evidence_packet,
    build_trusted_ai_inventory_evidence_packet,
    entry_signal_identity_sha256,
    verify_ai_inventory_evidence_packet,
    write_ai_inventory_evidence_packet,
    write_trusted_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory_consumer import (
    _ledger_tip as consumer_ledger_tip,
    _read_broker_ledger as consumer_read_broker_ledger,
)
from quant_rabbit.dojo_ai_source_capture import AiSourceCaptureError
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


def _dt(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, minute, second, tzinfo=timezone.utc)


def _source(
    source_id: str,
    digest: str,
    *,
    published: str = "2026-07-23T11:30:00Z",
    updated: str = "2026-07-23T11:35:00Z",
    fetched: str = "2026-07-23T11:40:00Z",
    observed: str = "2026-07-23T11:45:00Z",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_url": f"https://example.test/{source_id}",
        "title": f"Source {source_id}",
        "published_at_utc": published,
        "updated_at_utc": updated,
        "fetched_at_utc": fetched,
        "observed_at_utc": observed,
        "content_sha256": digest,
        "subject": "USD and JPY transmission context",
        "fact": "Confirmed point-in-time source fact.",
        "affected_currency": "USD",
        "transmission_chain": "Rates to USD to USD_JPY.",
        "observed_reaction": "Reaction observed before cutoff.",
        "contrary_evidence": "No independent contradiction in packet.",
        "confidence": 0.7,
    }


def _candle(
    completed: str,
    digest: str,
    *,
    started: str,
    granularity: str = "M1",
    max_age_seconds: int = 3_600,
) -> dict[str, object]:
    return {
        "pair": "USD_JPY",
        "granularity": granularity,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "bid_o": 163.1,
        "bid_h": 163.15,
        "bid_l": 163.05,
        "bid_c": 163.12,
        "ask_o": 163.11,
        "ask_h": 163.16,
        "ask_l": 163.06,
        "ask_c": 163.13,
        "source_sha256": digest,
        "max_age_seconds": max_age_seconds,
    }


def _packet() -> dict[str, object]:
    recent = "2026-07-23T11:59:30Z"
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": "2026-07-23T12:00:00Z",
        "bindings": {
            "launch_preflight_token_sha256": "a" * 64,
            "git_head": "b" * 40,
            "git_branch": "codex/test-ai-inventory",
            "canonical_source_root": DEDICATED_CANONICAL_SOURCE_ROOT.as_posix(),
            "experiment_id": "paper-ai-inventory-v1",
            "room_id": "paper-ai-inventory-room-01",
            "session_contract_sha256": "1" * 64,
            "candidate_id": "c" * 64,
            "candidate_sha256": "2" * 64,
            "spec_id": "candidate-spec-v1",
            "spec_sha256": "3" * 64,
            "policy_id": "inventory-policy-v1",
            "policy_sha256": "4" * 64,
            "paper_eligible_tip_sha256": "5" * 64,
            "ledger_sha256": "6" * 64,
            "ledger_observed_at_utc": recent,
            "state_sha256": "7" * 64,
            "state_observed_at_utc": recent,
            "snapshot_sha256": "8" * 64,
            "snapshot_observed_at_utc": recent,
        },
        "position": {
            "position_id": "T000001",
            "pair": "USD_JPY",
            "side": "LONG",
            "units": 2_000.5,
            "entry_price": 163.0,
            "opened_at_utc": "2026-07-23T11:10:00Z",
            "observed_at_utc": recent,
            "strategy_tag": "QR_DOJO_AI_INVENTORY_V1",
            "entry_context_sha256": "9" * 64,
            "take_profit": 163.3,
            "stop_loss": 162.75,
            "remaining_ceiling_seconds": 600,
            "unrealized_pl_jpy": 240.0,
            "gross_same_currency_units": 2_000,
            "net_same_currency_units": 2_000,
            "margin_used_jpy": 12_000.0,
            "capital_locked_jpy": 12_000.0,
            "same_direction_position_count": 1,
        },
        "entry_signal": None,
        "quote": {
            "pair": "USD_JPY",
            "bid": 163.12,
            "ask": 163.13,
            "timestamp_utc": recent,
            "source_sha256": "a" * 64,
            "max_age_seconds": 120,
        },
        # Deliberately reverse chronological; the builder must canonicalize.
        "candles": [
            _candle(
                "2026-07-23T11:59:00Z",
                "b" * 64,
                started="2026-07-23T11:58:00Z",
            ),
            _candle(
                "2026-07-23T11:58:00Z",
                "c" * 64,
                started="2026-07-23T11:57:00Z",
            ),
        ],
        "news_items": [
            _source("news-z", "d" * 64),
            _source(
                "news-a",
                "e" * 64,
                published="2026-07-23T11:00:00Z",
                updated="2026-07-23T11:05:00Z",
                fetched="2026-07-23T11:10:00Z",
                observed="2026-07-23T11:15:00Z",
            ),
        ],
        "calendar_items": [_source("calendar-us", "f" * 64)],
        "cross_asset_items": [_source("cross-us10y", "0" * 64)],
        "dynamic_binding_max_age_seconds": 120,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


def _entry_signal() -> dict[str, object]:
    signal: dict[str, object] = {
        "pair": "USD_JPY",
        "side": "LONG",
        "order_type": "MARKET",
        "units": 1_000.0,
        "price": None,
        "strategy_tag": "QR_DOJO_AI_INVENTORY_ENTRY_V1",
        "entry_context_sha256": "9" * 64,
        "tp_pips": 3.0,
        "sl_pips": 2.5,
        "observed_at_utc": "2026-07-23T11:59:30Z",
    }
    signal["signal_identity_sha256"] = entry_signal_identity_sha256(signal)
    return signal


def _flat_packet() -> dict[str, object]:
    packet = _packet()
    packet["position"] = {
        "position_id": "FLAT:USD_JPY",
        "pair": "USD_JPY",
        "side": "FLAT",
        "units": 0.0,
        "entry_price": None,
        "opened_at_utc": None,
        "observed_at_utc": "2026-07-23T11:59:30Z",
        "strategy_tag": "QR_DOJO_AI_INVENTORY_ENTRY_V1",
        "entry_context_sha256": "9" * 64,
        "take_profit": None,
        "stop_loss": None,
        "remaining_ceiling_seconds": 0,
        "unrealized_pl_jpy": 0.0,
        "gross_same_currency_units": 0.0,
        "net_same_currency_units": 0.0,
        "margin_used_jpy": 0.0,
        "capital_locked_jpy": 0.0,
        "same_direction_position_count": 0,
    }
    packet["entry_signal"] = _entry_signal()
    return packet


def _utc_ns(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000 + value.microsecond * 1_000


def _write_canonical_source(
    repository: Path,
    document: object,
    *,
    mtime: datetime | None = None,
) -> str:
    root = repository / DEDICATED_CANONICAL_SOURCE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path = root / f"{digest}.json"
    path.write_bytes(raw)
    timestamp = _utc_ns(mtime or _dt(23, 11, 59, 45))
    os.utime(path, ns=(timestamp, timestamp))
    return path.name


def _trusted_request(
    repository: Path,
    *,
    ledger_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    packet = _flat_packet()
    position = dict(packet["position"])
    position.pop("entry_context_sha256")
    entry_signal = dict(packet["entry_signal"])
    entry_signal.pop("signal_identity_sha256")
    entry_signal.pop("entry_context_sha256")
    quote = dict(packet["quote"])
    quote.pop("source_sha256")
    candles = []
    for item in packet["candles"]:
        row = dict(item)
        row.pop("source_sha256")
        candles.append(row)

    def without_content_sha(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        result = []
        for item in rows:
            row = dict(item)
            row.pop("content_sha256")
            result.append(row)
        return result

    observed = "2026-07-23T11:59:30Z"
    documents: dict[str, object] = {
        "session_contract": {
            "experiment_id": packet["bindings"]["experiment_id"],
            "room_id": packet["bindings"]["room_id"],
            "observed_at_utc": observed,
        },
        "candidate": {
            "candidate_id": packet["bindings"]["candidate_id"],
            "observed_at_utc": observed,
        },
        "spec": {
            "spec_id": packet["bindings"]["spec_id"],
            "observed_at_utc": observed,
        },
        "policy": {
            "policy_id": packet["bindings"]["policy_id"],
            "observed_at_utc": observed,
        },
        "paper_eligible_event": {
            "event_type": "PAPER_ELIGIBLE",
            "candidate_id": packet["bindings"]["candidate_id"],
            "event_sha256": "e" * 64,
            "observed_at_utc": observed,
        },
        "ledger": {
            "contract": "QR_VIRTUAL_BROKER_LEDGER_SNAPSHOT_V1",
            "room_id": packet["bindings"]["room_id"],
            "observed_at_utc": observed,
            "terminal_sha256": (ledger_rows[-1]["sha"] if ledger_rows else "0" * 64),
            "rows": ledger_rows or [],
        },
        "state": {
            "room_id": packet["bindings"]["room_id"],
            "observed_at_utc": observed,
            "status": "ACTIVE",
        },
        "snapshot": {
            "room_id": packet["bindings"]["room_id"],
            "observed_at_utc": observed,
            "positions": [],
        },
        "position": position,
        "entry_context": {
            "pair": packet["position"]["pair"],
            "strategy_tag": packet["position"]["strategy_tag"],
            "observed_at_utc": observed,
        },
        "entry_signal": entry_signal,
        "quote": quote,
        "candles": candles,
        "news_items": without_content_sha(packet["news_items"]),
        "calendar_items": without_content_sha(packet["calendar_items"]),
        "cross_asset_items": without_content_sha(packet["cross_asset_items"]),
    }
    source_files = {
        role: _write_canonical_source(repository, document)
        for role, document in documents.items()
    }
    source_receipts = {
        role: hashlib.sha256(f"capture:{role}".encode()).hexdigest()
        for role in source_files
    }
    return {
        "contract": DOJO_AI_EVIDENCE_PACKET_CONTRACT,
        "cutoff_utc": packet["cutoff_utc"],
        "experiment_id": packet["bindings"]["experiment_id"],
        "room_id": packet["bindings"]["room_id"],
        "candidate_id": packet["bindings"]["candidate_id"],
        "spec_id": packet["bindings"]["spec_id"],
        "policy_id": packet["bindings"]["policy_id"],
        "source_files": source_files,
        "source_receipts": source_receipts,
        "dynamic_binding_max_age_seconds": 120,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }


_TEST_GIT_HEAD = "b" * 40
_TEST_GIT_BRANCH = "codex/test-ai-inventory"


def _paper_eligible_preflight(
    request: dict[str, object],
    *,
    git_head: str = _TEST_GIT_HEAD,
) -> dict[str, object]:
    source_files = request["source_files"]
    body: dict[str, object] = {
        "contract": "QR_DOJO_AI_INVENTORY_LAUNCH_PREFLIGHT_V1",
        "candidate_id": request["candidate_id"],
        "adapter_id": "paper-ai-adapter-v1",
        "model_id": "paper-ai-model-v1",
        "config_sha256": "a" * 64,
        "producer_id": "paper-ai-producer-v1",
        "spec_sha256": Path(source_files["spec"]).stem,
        "policy_sha256": Path(source_files["policy"]).stem,
        "experiment_id": request["experiment_id"],
        "room_id": request["room_id"],
        "paper_eligible_event_sha256": "e" * 64,
        "candidate_lifecycle_ledger_tip_sha256": "e" * 64,
        "append_claim_sha256": "1" * 64,
        "job_manifest_sha256": "2" * 64,
        "job_owner_sha256": "3" * 64,
        "proof_artifact_sha256": "4" * 64,
        "proof_artifact_bytes_sha256": "5" * 64,
        "proof_manifest_sha256": "6" * 64,
        "replay_worker_receipt_sha256": "f" * 64,
        "source_manifest_sha256s": {"TRAIN": "7" * 64},
        "source_capture_manifest_sha256": "a" * 64,
        "future_registry_sha256": "8" * 64,
        "future_window": {
            "start_utc": "2026-07-23T11:00:00Z",
            "end_utc": "2026-07-23T13:00:00Z",
        },
        "git_head": git_head,
        "git_head_sha256": "9" * 64,
        "issued_at_utc": "2026-07-23T10:00:00Z",
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "paper_room_launched": False,
    }
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **body,
        "launch_preflight_token_sha256": hashlib.sha256(raw).hexdigest(),
    }


@contextmanager
def _trusted_runtime(
    repository: Path,
    request: dict[str, object],
    *,
    now: datetime | None = None,
    git_head: str = _TEST_GIT_HEAD,
    git_branch: str = _TEST_GIT_BRANCH,
):
    token = _paper_eligible_preflight(request, git_head=git_head)
    with (
        patch(
            "quant_rabbit.dojo_ai_evidence_packet._trusted_repository_root",
            return_value=repository,
        ),
        patch(
            "quant_rabbit.dojo_ai_evidence_packet._read_git_identity",
            return_value=(git_head, git_branch),
        ),
        patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=now or _dt(23, 12, 0, 30),
        ),
        patch(
            "quant_rabbit.dojo_ai_evidence_packet."
            "verify_paper_ai_inventory_launch_preflight",
            return_value=token,
        ),
        patch(
            "quant_rabbit.dojo_ai_evidence_packet."
            "verify_ai_source_capture_receipt",
            return_value={
                "contract": "QR_DOJO_AI_SOURCE_CAPTURE_RECEIPT_V1"
            },
        ),
    ):
        yield token


class DojoAiEvidencePacketTest(unittest.TestCase):
    def test_write_verify_is_content_addressed_and_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            with patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_dt(23, 12, 0, 30),
            ):
                path = write_ai_inventory_evidence_packet(repository, _packet())
                retry = write_ai_inventory_evidence_packet(repository, _packet())
            self.assertEqual(path, retry)
            self.assertEqual(path.parent, repository / DEDICATED_EVIDENCE_ROOT)
            packet = verify_ai_inventory_evidence_packet(repository, path)
            self.assertEqual(path.name, f"{packet['packet_sha256']}.json")
            self.assertEqual(
                [row["completed_at_utc"] for row in packet["candles"]],
                ["2026-07-23T11:58:00Z", "2026-07-23T11:59:00Z"],
            )
            self.assertEqual(
                [row["source_id"] for row in packet["news_items"]],
                ["news-a", "news-z"],
            )
            self.assertEqual(packet["position"]["units"], 2_000.5)
            self.assertNotIn("path", packet)
            self.assertTrue(packet["paper_only"])
            self.assertEqual(packet["order_authority"], "NONE")
            self.assertFalse(packet["live_permission"])

    def test_post_cutoff_candle_and_future_updated_source_fail(self) -> None:
        post_cutoff = _packet()
        post_cutoff["candles"][0]["completed_at_utc"] = "2026-07-23T12:00:01Z"
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "after immutable cutoff"):
                build_ai_inventory_evidence_packet(post_cutoff)

        future_update = _packet()
        future_update["news_items"][0]["updated_at_utc"] = "2026-07-23T12:00:01Z"
        future_update["news_items"][0]["fetched_at_utc"] = "2026-07-23T12:00:01Z"
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "after immutable cutoff"):
                build_ai_inventory_evidence_packet(future_update)

    def test_tamper_is_rejected_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            with patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_dt(23, 12, 0, 30),
            ):
                path = write_ai_inventory_evidence_packet(repository, _packet())
            payload = json.loads(path.read_text())
            payload["quote"]["bid"] = 1.0
            path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            tampered = path.read_bytes()
            with self.assertRaises(EvidencePacketIntegrityError):
                verify_ai_inventory_evidence_packet(repository, path)
            with patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_dt(23, 12, 0, 30),
            ):
                with self.assertRaises(EvidencePacketIntegrityError):
                    write_ai_inventory_evidence_packet(repository, _packet())
            self.assertEqual(path.read_bytes(), tampered)

    def test_weekend_cutoff_and_weekend_writer_clock_fail_closed(self) -> None:
        weekend_cutoff = _packet()
        weekend_cutoff["cutoff_utc"] = "2026-07-25T12:00:00Z"
        weekend_cutoff["quote"]["timestamp_utc"] = "2026-07-25T11:59:30Z"
        for key in (
            "ledger_observed_at_utc",
            "state_observed_at_utc",
            "snapshot_observed_at_utc",
        ):
            weekend_cutoff["bindings"][key] = "2026-07-25T11:59:30Z"
        weekend_cutoff["position"]["observed_at_utc"] = "2026-07-25T11:59:30Z"
        weekend_cutoff["candles"] = [
            _candle(
                "2026-07-25T11:59:00Z",
                "b" * 64,
                started="2026-07-25T11:58:00Z",
            )
        ]
        weekend_cutoff["news_items"] = []
        weekend_cutoff["calendar_items"] = []
        weekend_cutoff["cross_asset_items"] = []
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(25, 12, 0, 30),
        ):
            with self.assertRaises(EvidencePacketMarketClosedError):
                build_ai_inventory_evidence_packet(weekend_cutoff)

        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(25, 12, 0, 30),
        ):
            with self.assertRaises(EvidencePacketMarketClosedError):
                build_ai_inventory_evidence_packet(_packet())

    def test_stale_quote_and_candle_fail_closed(self) -> None:
        packet = _packet()
        packet["quote"]["timestamp_utc"] = "2026-07-23T11:57:00Z"
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "quote.timestamp_utc is stale"):
                build_ai_inventory_evidence_packet(packet)

        stale_candle = _packet()
        stale_candle["candles"] = [
            _candle(
                "2026-07-23T11:00:00Z",
                "b" * 64,
                started="2026-07-23T10:59:00Z",
                max_age_seconds=120,
            )
        ]
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "completed_at_utc is stale"):
                build_ai_inventory_evidence_packet(stale_candle)

    def test_duplicate_source_rejected_while_unsorted_source_is_sorted(self) -> None:
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            built = build_ai_inventory_evidence_packet(_packet())
        self.assertEqual(
            [row["source_id"] for row in built["news_items"]],
            ["news-a", "news-z"],
        )

        duplicate = _packet()
        duplicate["news_items"].append(_source("news-z", "1" * 64))
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate news source"):
                build_ai_inventory_evidence_packet(duplicate)

    def test_flat_entry_admission_packet_is_strict_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            with patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_dt(23, 12, 0, 30),
            ):
                path = write_ai_inventory_evidence_packet(repository, _flat_packet())
            packet = verify_ai_inventory_evidence_packet(repository, path)
            self.assertEqual(packet["position"]["side"], "FLAT")
            self.assertEqual(packet["position"]["position_id"], "FLAT:USD_JPY")
            self.assertIsInstance(packet["position"]["units"], float)
            self.assertEqual(packet["position"]["units"], 0.0)
            self.assertIsNone(packet["position"]["entry_price"])
            self.assertIsNone(packet["position"]["opened_at_utc"])
            self.assertEqual(packet["position"]["same_direction_position_count"], 0)
            self.assertEqual(
                packet["entry_signal"]["signal_identity_sha256"],
                entry_signal_identity_sha256(packet["entry_signal"]),
            )
            self.assertEqual(packet["bindings"]["paper_eligible_tip_sha256"], "5" * 64)

    def test_flat_nonzero_and_ambiguous_values_fail_closed(self) -> None:
        cases = (
            ("position_id", "FLAT:EUR_USD", "FLAT:<pair>"),
            ("units", 0, "exact float zero"),
            ("units", 1.0, "exact float zero"),
            ("entry_price", 0.0, "must be null"),
            ("opened_at_utc", "2026-07-23T11:00:00Z", "must be null"),
            ("unrealized_pl_jpy", 1.0, "exact float zero"),
            ("gross_same_currency_units", 0, "exact float zero"),
            ("remaining_ceiling_seconds", 1, "exact integer zero"),
            ("remaining_ceiling_seconds", 0.0, "exact integer zero"),
            ("same_direction_position_count", 1, "exact integer zero"),
            ("same_direction_position_count", 0.0, "exact integer zero"),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field, invalid=invalid):
                packet = _flat_packet()
                packet["position"][field] = invalid
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                    return_value=_dt(23, 12, 0, 30),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        build_ai_inventory_evidence_packet(packet)

    def test_open_position_accepts_fractional_units_but_rejects_nulls(self) -> None:
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            built = build_ai_inventory_evidence_packet(_packet())
        self.assertEqual(built["position"]["units"], 2_000.5)
        self.assertIsInstance(built["position"]["units"], float)

        for field in ("entry_price", "opened_at_utc", "take_profit", "stop_loss"):
            with self.subTest(field=field):
                packet = _packet()
                packet["position"][field] = None
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                    return_value=_dt(23, 12, 0, 30),
                ):
                    with self.assertRaises(ValueError):
                        build_ai_inventory_evidence_packet(packet)

    def test_flat_requires_one_digest_bound_entry_signal(self) -> None:
        missing = _flat_packet()
        missing["entry_signal"] = None
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(TypeError, "entry_signal must be an object"):
                build_ai_inventory_evidence_packet(missing)

        tampered = _flat_packet()
        tampered["entry_signal"]["units"] = 2_000.0
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "identity_sha256 mismatch"):
                build_ai_inventory_evidence_packet(tampered)

    def test_entry_signal_scope_mismatch_fails_closed(self) -> None:
        cases = (
            ("pair", "EUR_USD", "pair does not match"),
            ("strategy_tag", "OTHER_STRATEGY", "strategy_tag does not match"),
            (
                "entry_context_sha256",
                "1" * 64,
                "entry_context_sha256 does not match",
            ),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field):
                packet = _flat_packet()
                packet["entry_signal"][field] = invalid
                packet["entry_signal"]["signal_identity_sha256"] = (
                    entry_signal_identity_sha256(packet["entry_signal"])
                )
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                    return_value=_dt(23, 12, 0, 30),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        build_ai_inventory_evidence_packet(packet)

    def test_entry_signal_future_and_stale_timestamps_fail_closed(self) -> None:
        for observed, message in (
            ("2026-07-23T12:00:01Z", "after immutable cutoff"),
            ("2026-07-23T11:57:00Z", "is stale"),
        ):
            with self.subTest(observed=observed):
                packet = _flat_packet()
                packet["entry_signal"]["observed_at_utc"] = observed
                packet["entry_signal"]["signal_identity_sha256"] = (
                    entry_signal_identity_sha256(packet["entry_signal"])
                )
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                    return_value=_dt(23, 12, 0, 30),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        build_ai_inventory_evidence_packet(packet)

    def test_entry_signal_pip_distances_are_optional_and_canonical(self) -> None:
        packet = _flat_packet()
        packet["entry_signal"]["tp_pips"] = None
        packet["entry_signal"]["sl_pips"] = None
        packet["entry_signal"]["signal_identity_sha256"] = entry_signal_identity_sha256(
            packet["entry_signal"]
        )
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            built = build_ai_inventory_evidence_packet(packet)
        self.assertIsNone(built["entry_signal"]["tp_pips"])
        self.assertIsNone(built["entry_signal"]["sl_pips"])

        for field, invalid in (
            ("tp_pips", 0),
            ("tp_pips", -1.0),
            ("sl_pips", 0.0),
            ("sl_pips", -0.25),
        ):
            with self.subTest(field=field, invalid=invalid):
                packet = _flat_packet()
                packet["entry_signal"][field] = invalid
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    entry_signal_identity_sha256(packet["entry_signal"])

    def test_entry_signal_numeric_identity_normalizes_int_and_float(self) -> None:
        float_signal = _entry_signal()
        int_signal = dict(float_signal)
        int_signal["units"] = 1_000
        int_signal["tp_pips"] = 3
        self.assertEqual(
            entry_signal_identity_sha256(float_signal),
            entry_signal_identity_sha256(int_signal),
        )

        float_limit = _entry_signal()
        float_limit["order_type"] = "LIMIT"
        float_limit["price"] = 163.0
        int_limit = dict(float_limit)
        int_limit["price"] = 163
        int_limit["units"] = 1_000
        int_limit["tp_pips"] = 3
        self.assertEqual(
            entry_signal_identity_sha256(float_limit),
            entry_signal_identity_sha256(int_limit),
        )
        int_limit["signal_identity_sha256"] = entry_signal_identity_sha256(int_limit)
        packet = _flat_packet()
        packet["entry_signal"] = int_limit
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            built = build_ai_inventory_evidence_packet(packet)
        self.assertIsInstance(built["entry_signal"]["units"], float)
        self.assertIsInstance(built["entry_signal"]["price"], float)
        self.assertIsInstance(built["entry_signal"]["tp_pips"], float)
        self.assertIsInstance(built["entry_signal"]["sl_pips"], float)

    def test_open_position_rejects_non_null_entry_signal(self) -> None:
        packet = _packet()
        packet["entry_signal"] = _entry_signal()
        with patch(
            "quant_rabbit.dojo_ai_evidence_packet._utc_now",
            return_value=_dt(23, 12, 0, 30),
        ):
            with self.assertRaisesRegex(ValueError, "must be null"):
                build_ai_inventory_evidence_packet(packet)

    def test_trusted_builder_derives_all_digests_from_canonical_files(self) -> None:
        self.assertFalse(LOW_LEVEL_BUILDER_LAUNCH_SAFE)
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            with _trusted_runtime(repository, request) as token:
                packet = build_trusted_ai_inventory_evidence_packet(request)
                path = write_trusted_ai_inventory_evidence_packet(request)
            verified = verify_ai_inventory_evidence_packet(repository, path)
            self.assertEqual(packet, verified)
            self.assertNotIn("sha256", request)
            quote_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["quote"]
            )
            self.assertEqual(
                packet["quote"]["source_sha256"],
                hashlib.sha256(quote_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                packet["position"]["entry_context_sha256"],
                Path(request["source_files"]["entry_context"]).stem,
            )
            self.assertEqual(
                packet["entry_signal"]["signal_identity_sha256"],
                entry_signal_identity_sha256(packet["entry_signal"]),
            )
            ledger_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["ledger"]
            )
            self.assertEqual(packet["bindings"]["ledger_sha256"], "0" * 64)
            self.assertNotEqual(
                packet["bindings"]["ledger_sha256"],
                hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            )
            self.assertNotIn("ledger_source_sha256", packet["bindings"])
            self.assertEqual(
                packet["bindings"]["launch_preflight_token_sha256"],
                token["launch_preflight_token_sha256"],
            )
            self.assertEqual(packet["bindings"]["git_head"], _TEST_GIT_HEAD)
            self.assertEqual(packet["bindings"]["git_branch"], _TEST_GIT_BRANCH)
            self.assertEqual(
                packet["bindings"]["canonical_source_root"],
                DEDICATED_CANONICAL_SOURCE_ROOT.as_posix(),
            )

    def test_real_virtual_broker_ledger_tip_roundtrips_to_consumer_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            broker_path = repository / "virtual-broker.jsonl"
            broker = VirtualBroker(broker_path)
            broker.on_quote(
                "USD_JPY",
                163.10,
                163.11,
                "2026-07-23T11:59:20Z",
            )
            with patch("quant_rabbit.virtual_broker.datetime") as broker_clock:
                broker_clock.now.return_value = _dt(23, 11, 59, 30)
                with self.assertRaises(VirtualBrokerError):
                    broker.market_order("USD_JPY", "LONG", 10**12)
            broker_tip = broker._prev_sha
            broker._handle.close()
            ledger_rows = [
                json.loads(line)
                for line in broker_path.read_text(encoding="utf-8").splitlines()
            ]
            request = _trusted_request(repository, ledger_rows=ledger_rows)
            with _trusted_runtime(repository, request):
                packet = build_trusted_ai_inventory_evidence_packet(request)

            consumer_rows = consumer_read_broker_ledger(broker_path)
            self.assertEqual(consumer_ledger_tip(consumer_rows), broker_tip)
            self.assertEqual(packet["bindings"]["ledger_sha256"], broker_tip)
            self.assertNotIn("ledger_source_sha256", packet["bindings"])

    def test_trusted_virtual_broker_ledger_rejects_chain_and_terminal_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            body = {
                "ts_utc": "2026-07-23T11:59:00Z",
                "event": "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                "payload": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 1_000_000_000,
                },
                "prev_sha": "0" * 64,
            }
            row = {
                **body,
                "sha": hashlib.sha256(
                    json.dumps(
                        body,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            request = _trusted_request(repository, ledger_rows=[row])
            ledger_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["ledger"]
            )
            ledger = json.loads(ledger_path.read_text())
            ledger["rows"][0]["payload"]["units"] += 1
            request["source_files"]["ledger"] = _write_canonical_source(
                repository, ledger
            )
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError, "ledger sha mismatch"
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

            request = _trusted_request(repository)
            ledger_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["ledger"]
            )
            ledger = json.loads(ledger_path.read_text())
            ledger["terminal_sha256"] = "f" * 64
            request["source_files"]["ledger"] = _write_canonical_source(
                repository, ledger
            )
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError,
                    "terminal_sha256 mismatch",
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

    def test_trusted_api_has_no_caller_repository_root_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            with self.assertRaises(TypeError):
                build_trusted_ai_inventory_evidence_packet(
                    repository,
                    request,  # type: ignore[call-arg]
                )
            with self.assertRaises(TypeError):
                write_trusted_ai_inventory_evidence_packet(
                    repository,
                    request,  # type: ignore[call-arg]
                )
            with self.assertRaises(TypeError):
                build_trusted_ai_inventory_evidence_packet(
                    request,
                    launch_preflight_token={},  # type: ignore[call-arg]
                )

    def test_paper_eligible_preflight_rejects_schema_confusion_and_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            with _trusted_runtime(repository, request) as token:
                generic_same_contract_token = {
                    "contract": "QR_DOJO_AI_INVENTORY_LAUNCH_PREFLIGHT_V1",
                    "repository_root": str(repository),
                    "git_head": _TEST_GIT_HEAD,
                    "git_branch": _TEST_GIT_BRANCH,
                    "canonical_source_root": str(
                        repository / DEDICATED_CANONICAL_SOURCE_ROOT
                    ),
                    "evidence_root": str(repository / DEDICATED_EVIDENCE_ROOT),
                    "sealed_at_utc": "2026-07-23T12:00:00Z",
                    "expires_at_utc": "2026-07-23T12:05:00Z",
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                    "token_sha256": "0" * 64,
                }
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet."
                    "verify_paper_ai_inventory_launch_preflight",
                    return_value=generic_same_contract_token,
                ):
                    with self.assertRaisesRegex(
                        EvidencePacketIntegrityError, "schema is invalid"
                    ):
                        build_trusted_ai_inventory_evidence_packet(request)

                tampered = dict(token)
                tampered["policy_sha256"] = "f" * 64
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet."
                    "verify_paper_ai_inventory_launch_preflight",
                    return_value=tampered,
                ):
                    with self.assertRaisesRegex(
                        EvidencePacketIntegrityError, "digest mismatch"
                    ):
                        build_trusted_ai_inventory_evidence_packet(request)

                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._read_git_identity",
                    return_value=("c" * 40, _TEST_GIT_BRANCH),
                ):
                    with self.assertRaisesRegex(
                        EvidencePacketIntegrityError,
                        "Git HEAD no longer matches",
                    ):
                        build_trusted_ai_inventory_evidence_packet(request)

                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet._read_git_identity",
                    return_value=(_TEST_GIT_HEAD, "main"),
                ):
                    with self.assertRaisesRegex(
                        EvidencePacketIntegrityError,
                        "runtime Git identity",
                    ):
                        build_trusted_ai_inventory_evidence_packet(request)

    def test_evidence_cutoff_must_be_inside_preflight_future_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            request["cutoff_utc"] = "2026-07-23T13:00:00Z"
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError, "outside PAPER_ELIGIBLE"
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

    def test_trusted_builder_rejects_external_path_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            request["source_files"]["quote"] = "/tmp/not-canonical.json"
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError, "content-addressed"
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

            request = _trusted_request(repository)
            quote_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["quote"]
            )
            outside = repository / "outside.json"
            outside.write_bytes(quote_path.read_bytes())
            quote_path.unlink()
            quote_path.symlink_to(outside)
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError, "non-symlink"
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

    def test_trusted_builder_rejects_digest_mismatch_and_self_claimed_sha(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            quote_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["quote"]
            )
            quote = json.loads(quote_path.read_text())
            quote["bid"] = 162.0
            raw = (
                json.dumps(quote, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
            quote_path.write_bytes(raw)
            timestamp = _utc_ns(_dt(23, 11, 59, 45))
            os.utime(quote_path, ns=(timestamp, timestamp))
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(
                    EvidencePacketIntegrityError, "digest does not match"
                ):
                    build_trusted_ai_inventory_evidence_packet(request)

            request = _trusted_request(repository)
            quote_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["quote"]
            )
            quote = json.loads(quote_path.read_text())
            quote["source_sha256"] = "f" * 64
            request["source_files"]["quote"] = _write_canonical_source(
                repository, quote
            )
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(ValueError, "trusted quote schema"):
                    build_trusted_ai_inventory_evidence_packet(request)

    def test_trusted_builder_rejects_post_cutoff_rows_but_not_future_mtime(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            candle_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["candles"]
            )
            candles = json.loads(candle_path.read_text())
            candles[0]["completed_at_utc"] = "2026-07-23T12:00:01Z"
            request["source_files"]["candles"] = _write_canonical_source(
                repository, candles
            )
            with _trusted_runtime(repository, request):
                with self.assertRaisesRegex(ValueError, "after immutable cutoff"):
                    build_trusted_ai_inventory_evidence_packet(request)

            request = _trusted_request(repository)
            state_path = (
                repository
                / DEDICATED_CANONICAL_SOURCE_ROOT
                / request["source_files"]["state"]
            )
            future_ns = _utc_ns(_dt(23, 12, 0, 1))
            os.utime(state_path, ns=(future_ns, future_ns))
            with _trusted_runtime(repository, request):
                packet = build_trusted_ai_inventory_evidence_packet(request)
            self.assertEqual(
                packet["bindings"]["state_sha256"],
                Path(request["source_files"]["state"]).stem,
            )

    def test_trusted_builder_requires_every_signed_capture_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            request.pop("source_receipts")
            with self.assertRaisesRegex(
                ValueError, "trusted evidence request schema"
            ):
                build_trusted_ai_inventory_evidence_packet(request)

            request = _trusted_request(repository)
            with _trusted_runtime(repository, request):
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet."
                    "verify_ai_source_capture_receipt",
                    side_effect=AiSourceCaptureError("receipt absent"),
                ):
                    with self.assertRaisesRegex(
                        EvidencePacketIntegrityError,
                        "lacks a valid signed acquisition receipt",
                    ):
                        build_trusted_ai_inventory_evidence_packet(request)

    def test_trusted_builder_checks_receipt_for_every_source_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            with _trusted_runtime(repository, request):
                with patch(
                    "quant_rabbit.dojo_ai_evidence_packet."
                    "verify_ai_source_capture_receipt",
                    return_value={
                        "contract": "QR_DOJO_AI_SOURCE_CAPTURE_RECEIPT_V1"
                    },
                ) as verifier:
                    build_trusted_ai_inventory_evidence_packet(request)
            self.assertEqual(
                verifier.call_count,
                len(request["source_files"]),
            )
            checked_roles = {
                call.kwargs["source_role"] for call in verifier.call_args_list
            }
            self.assertEqual(checked_roles, set(request["source_files"]))

    def test_trusted_builder_rejects_duplicate_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            request = _trusted_request(repository)
            root = repository / DEDICATED_CANONICAL_SOURCE_ROOT
            timestamp = _utc_ns(_dt(23, 11, 59, 45))
            for raw, message in (
                (
                    b'{"bid":163.12,"bid":163.11,"pair":"USD_JPY"}\n',
                    "JSON is invalid",
                ),
                (
                    b'{"bid":NaN,"pair":"USD_JPY"}\n',
                    "JSON is invalid",
                ),
            ):
                with self.subTest(raw=raw):
                    digest = hashlib.sha256(raw).hexdigest()
                    path = root / f"{digest}.json"
                    path.write_bytes(raw)
                    os.utime(path, ns=(timestamp, timestamp))
                    request["source_files"]["quote"] = path.name
                    with _trusted_runtime(repository, request):
                        with self.assertRaisesRegex(
                            EvidencePacketIntegrityError, message
                        ):
                            build_trusted_ai_inventory_evidence_packet(request)

    def test_verifier_rejects_dedicated_root_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary).resolve()
            with patch(
                "quant_rabbit.dojo_ai_evidence_packet._utc_now",
                return_value=_dt(23, 12, 0, 30),
            ):
                path = write_ai_inventory_evidence_packet(repository, _packet())
            outside = repository / path.name
            outside.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(
                EvidencePacketIntegrityError, "escapes the dedicated root"
            ):
                verify_ai_inventory_evidence_packet(repository, outside)


if __name__ == "__main__":
    unittest.main()
