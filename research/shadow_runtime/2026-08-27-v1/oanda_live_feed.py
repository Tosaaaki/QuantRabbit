"""GET-only OANDA v20 LIVE pricing recorder for zero-order shadow input."""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
import signal
import ssl
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from shadow_runtime import (
    HashLedger,
    IntegrityError,
    RuntimeLock,
    atomic_json,
    canonical_hash,
    parse_utc,
    real_dir,
    secure_read,
    utc_text,
)

PROVIDER = "OANDA_V20_LIVE_PRICING_STREAM"
REST_HOST = "https://api-fxtrade.oanda.com"
STREAM_HOST = "https://stream-fxtrade.oanda.com"
REST_NETLOC = "api-fxtrade.oanda.com"
STREAM_NETLOC = "stream-fxtrade.oanda.com"
APPROVED_ENV_FILE = Path("/Users/tossaki/App/QuantRabbit-live/.env.local")
SYMBOLS = ("EUR_USD", "USD_JPY")
CONTINUITY = "HEARTBEAT_ONLY"
LOSSLESS = False
MAX_HEARTBEAT_GAP_SECONDS = 15.0
LEDGERS = (
    "raw_bbo",
    "historical_warmup_m5",
    "feed_quality",
    "decisions",
    "virtual_fills",
    "pnl",
    "control",
)
LEGACY_SEGMENT_ID = "LEGACY_UNSEGMENTED"


class FeedQualityError(RuntimeError):
    pass


def _clean_env_value(value: str) -> str:
    text = value.strip()
    if "#" in text and not (text.startswith('"') or text.startswith("'")):
        text = text.split("#", 1)[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text


def load_approved_live_credentials() -> tuple[str, str]:
    path = APPROVED_ENV_FILE
    before = os.lstat(path)
    if not path.is_file() or path.is_symlink() or before.st_nlink != 1:
        raise PermissionError("APPROVED_ENV_FILE_UNSAFE")
    if before.st_uid != os.getuid() or before.st_mode & 0o077:
        raise PermissionError("APPROVED_ENV_FILE_PERMISSION_MISMATCH")
    values: dict[str, str] = {}
    for raw_line in secure_read(path).decode("utf-8", "strict").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"QR_OANDA_TOKEN", "QR_OANDA_ACCOUNT_ID", "QR_OANDA_BASE_URL"}:
            if key in values:
                raise PermissionError("DUPLICATE_APPROVED_CREDENTIAL_KEY")
            values[key] = _clean_env_value(value)
    if values.get("QR_OANDA_BASE_URL") != REST_HOST:
        raise PermissionError("OANDA_LIVE_HOST_NOT_EXACT")
    token = values.get("QR_OANDA_TOKEN", "")
    account_id = values.get("QR_OANDA_ACCOUNT_ID", "")
    if not token or not account_id:
        raise PermissionError("OANDA_LIVE_CREDENTIAL_MISSING")
    return account_id, token


def parse_oanda_time(value: str) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, fraction = core.split(".", 1)
            text = f"{head}.{fraction[:6]}+00:00"
        else:
            text = f"{core}+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _market_event_digest(event: dict[str, Any]) -> str:
    """Bind the market fact while allowing an identical replay after reconnect."""
    transport_keys = {
        "arrival_time_utc", "segment_id", "segment_started_at_utc",
        "feed_service_attestation_hash", "feed_provenance_status",
    }
    return canonical_hash({key: value for key, value in event.items() if key not in transport_keys})


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class OandaLiveRecorder:
    def __init__(self, runtime_root: Path):
        self.runtime_root = Path(runtime_root)
        self.ledger_root = self.runtime_root / "ledgers"
        real_dir(self.runtime_root)
        real_dir(self.ledger_root)
        self.contract = json.loads(
            secure_read(Path(__file__).resolve().parent / "oanda_live_runtime_contract.json")
        )
        self.contract_hash = canonical_hash(self.contract)
        self._verify_contract()
        self.state_path = self.runtime_root / "state.json"
        self.checkpoint_path = self.runtime_root / "checkpoint.json"
        self.status_path = self.runtime_root / "status.json"
        self.ledgers = {
            name: HashLedger(self.ledger_root / f"{name}.jsonl") for name in LEDGERS
        }
        self.state = self._load_state()
        self.stop_requested = False

    def _verify_contract(self) -> None:
        if self.contract["provider"] != PROVIDER:
            raise IntegrityError("provider mismatch")
        if self.contract["rest_host"] != REST_HOST or self.contract["stream_host"] != STREAM_HOST:
            raise IntegrityError("host mismatch")
        if tuple(self.contract["symbols"]) != SYMBOLS or self.contract["fallback_providers"] != []:
            raise IntegrityError("symbol/fallback mismatch")
        if self.contract["http_method_allowlist"] != ["GET"]:
            raise IntegrityError("method allowlist mismatch")
        if self.contract["live_order_authority"] or self.contract["external_orders"] != 0:
            raise IntegrityError("order authority mismatch")
        if self.contract["bot_only"] or not self.contract["actual_llm_enabled"]:
            raise IntegrityError("arm boundary mismatch")
        paper = self.contract.get("paper_execution")
        llm_policy = self.contract.get("llm_inventory_policy")
        if (
            not isinstance(paper, dict)
            or paper.get("enabled") is not True
            or paper.get("entry_cost_gate_used") is not False
            or paper.get("external_order_authority") is not False
            or not isinstance(llm_policy, dict)
            or llm_policy.get("enabled") is not True
            or llm_policy.get("individual_order_control") is not False
            or llm_policy.get("external_order_authority") is not False
        ):
            raise IntegrityError("paper/LLM authority boundary mismatch")
        warmup = self.contract.get("historical_warmup")
        if (
            not isinstance(warmup, dict)
            or warmup.get("provider") != "OANDA_V20_LIVE_HOST_HISTORICAL_CANDLES"
            or warmup.get("timeframe") != "M5"
            or warmup.get("price") != "BA"
            or type(warmup.get("request_count")) is not int
            or not 1 <= warmup["request_count"] <= 5000
            or warmup.get("completed_only") is not True
            or warmup.get("strict_contiguous") is not True
            or warmup.get("excluded_from_forward_pnl") is not True
            or warmup.get("may_create_proposals_fills_or_pnl") is not False
        ):
            raise IntegrityError("historical warmup contract mismatch")

    def fresh_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": PROVIDER,
            "contract_hash": self.contract_hash,
            "host": STREAM_HOST,
            "symbols": list(SYMBOLS),
            "continuity": CONTINUITY,
            "lossless": LOSSLESS,
            "run_state": "READY",
            "feed_connected": False,
            "connection_established": False,
            "segment_id": None,
            "segment_started_at_utc": None,
            "feed_service_attestation_hash": None,
            "feed_provenance_status": "LEGACY_MIGRATION_UNATTESTED",
            "segment_heartbeats": 0,
            "fresh_symbols": [],
            "feed_blocked": False,
            "block_reason": None,
            "last_arrival_utc": None,
            "last_source_time": {},
            "seen_event_ids": {},
            "counters": {
                "network_attempts": 0,
                "credential_reads": 0,
                "credential_values_persisted": 0,
                "connections": 0,
                "segments": 0,
                "heartbeats": 0,
                "duplicate_heartbeats": 0,
                "market_events_received": 0,
                "market_events_accepted": 0,
                "duplicate_events": 0,
                "gaps": 0,
                "out_of_order": 0,
                "malformed": 0,
                "decisions": 0,
                "virtual_fills": 0,
                "pnl_records": 0,
                "llm_calls": 0,
                "external_order_attempts": 0,
                "external_orders": 0
            }
        }

    def _checkpoint(self, state: dict[str, Any]) -> dict[str, Any]:
        row = {
            "schema_version": 1,
            "contract_hash": self.contract_hash,
            "state_hash": canonical_hash(state),
            "ledger_heads": {name: ledger.last_hash for name, ledger in self.ledgers.items()},
            "ledger_rows": {name: len(ledger.rows) for name, ledger in self.ledgers.items()}
        }
        row["checkpoint_hash"] = canonical_hash(row)
        return row

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists() and not self.checkpoint_path.exists():
            return self.fresh_state()
        if not self.state_path.exists() or not self.checkpoint_path.exists():
            raise IntegrityError("state/checkpoint pair incomplete")
        state = json.loads(secure_read(self.state_path).decode("utf-8", "strict"))
        checkpoint = json.loads(secure_read(self.checkpoint_path).decode("utf-8", "strict"))
        if checkpoint != self._checkpoint(state):
            raise IntegrityError("checkpoint mismatch")
        return state

    def _persist(self, state: dict[str, Any]) -> None:
        atomic_json(self.state_path, state)
        atomic_json(self.checkpoint_path, self._checkpoint(state))
        self.state = state
        atomic_json(self.status_path, self.status())

    def _append(self, ledger: str, payload: dict[str, Any], record_id: str) -> dict[str, Any]:
        planned: list[dict[str, Any]] = []
        row = self.ledgers[ledger].plan(payload, record_id, planned)
        self.ledgers[ledger].append_rows(planned)
        return row

    def connect_started(self, service_attestation_hash: str) -> None:
        if not valid_sha256(service_attestation_hash):
            raise IntegrityError("FEED_SERVICE_ATTESTATION_INVALID")
        state = copy.deepcopy(self.state)
        state["counters"].setdefault("segments", 0)
        state["counters"]["segments"] += 1
        state.update(
            run_state="CONNECTING",
            feed_connected=False,
            feed_blocked=False,
            block_reason=None,
            segment_id=f"segment-{state['counters']['segments']:08d}",
            segment_started_at_utc=None,
            feed_service_attestation_hash=service_attestation_hash,
            feed_provenance_status="ATTESTED",
            segment_heartbeats=0,
            fresh_symbols=[],
            last_arrival_utc=None,
            last_source_time={},
            seen_event_ids={},
        )
        state["counters"]["network_attempts"] += 1
        self._persist(state)

    def mark_approved_credential_file_read(self) -> None:
        state = copy.deepcopy(self.state)
        state["counters"]["credential_reads"] += 1
        self._persist(state)

    def mark_network_attempt(self) -> None:
        state = copy.deepcopy(self.state)
        state["counters"]["network_attempts"] += 1
        self._persist(state)

    def historical_warmup_ready(self, instrument: str, expected_count: int) -> bool:
        """Return true only for one complete, reusable warmup prefix."""
        rows = [
            row["payload"]
            for row in self.ledgers["historical_warmup_m5"].rows
            if row["payload"].get("instrument") == instrument
        ]
        if not rows:
            return False
        if len(rows) != expected_count:
            raise IntegrityError("HISTORICAL_WARMUP_PARTIAL")
        rows.sort(key=lambda payload: payload["start_utc"])
        starts = [parse_utc(payload["start_utc"]) for payload in rows]
        if any(current - prior != timedelta(minutes=5) for prior, current in zip(starts, starts[1:])):
            raise IntegrityError("HISTORICAL_WARMUP_GAP")
        if any(
            payload.get("feature_source") != "OANDA_HISTORICAL_M5_WARMUP"
            or payload.get("warmup_only") is not True
            or payload.get("excluded_from_forward_pnl") is not True
            or payload.get("proposals") != 0
            or payload.get("virtual_fills") != 0
            or payload.get("pnl_records") != 0
            or payload.get("external_order_attempts") != 0
            or payload.get("external_orders") != 0
            for payload in rows
        ):
            raise IntegrityError("HISTORICAL_WARMUP_BOUNDARY_INVALID")
        return True

    def _historical_warmup_payload(
        self,
        *,
        instrument: str,
        fetched_at: datetime,
        response_sha256: str,
        request_sha256: str,
        candle: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if instrument not in SYMBOLS or not valid_sha256(response_sha256) or not valid_sha256(request_sha256):
            raise IntegrityError("HISTORICAL_WARMUP_IDENTITY_INVALID")
        start = parse_oanda_time(str(candle["time"]))
        end = start + timedelta(minutes=5)
        if candle.get("complete") is not True or end > fetched_at:
            raise IntegrityError("HISTORICAL_WARMUP_NONCAUSAL")
        bid = candle.get("bid")
        ask = candle.get("ask")
        if not isinstance(bid, dict) or not isinstance(ask, dict):
            raise IntegrityError("HISTORICAL_WARMUP_BID_ASK_MISSING")
        normalized: dict[str, float] = {}
        for side, body in (("bid", bid), ("ask", ask)):
            for field in ("o", "h", "l", "c"):
                try:
                    value = float(body[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise IntegrityError("HISTORICAL_WARMUP_PRICE_INVALID") from exc
                if not math.isfinite(value) or value <= 0:
                    raise IntegrityError("HISTORICAL_WARMUP_PRICE_INVALID")
                normalized[f"{side}_{field}"] = value
            if not (
                normalized[f"{side}_l"]
                <= min(normalized[f"{side}_o"], normalized[f"{side}_c"])
                <= max(normalized[f"{side}_o"], normalized[f"{side}_c"])
                <= normalized[f"{side}_h"]
            ):
                raise IntegrityError("HISTORICAL_WARMUP_OHLC_INVALID")
        if normalized["bid_o"] > normalized["ask_o"] or normalized["bid_c"] > normalized["ask_c"]:
            raise IntegrityError("HISTORICAL_WARMUP_BID_ASK_CROSSED")
        candle_input = {
            "instrument": instrument,
            "time": utc_text(start),
            "complete": True,
            "bid": {key: normalized[f"bid_{key}"] for key in ("o", "h", "l", "c")},
            "ask": {key: normalized[f"ask_{key}"] for key in ("o", "h", "l", "c")},
        }
        input_sha256 = canonical_hash(candle_input)
        payload = {
            "schema_version": 1,
            "event": "HISTORICAL_M5_WARMUP",
            "instrument": instrument,
            "timeframe": "M5",
            "feature_source": "OANDA_HISTORICAL_M5_WARMUP",
            "start_utc": utc_text(start),
            "end_utc": utc_text(end),
            "source_time_utc": utc_text(start),
            "arrival_time_utc": utc_text(fetched_at),
            "response_sha256": response_sha256,
            "request_sha256": request_sha256,
            "input_sha256": input_sha256,
            **normalized,
            "warmup_only": True,
            "excluded_from_forward_pnl": True,
            "proposals": 0,
            "virtual_fills": 0,
            "pnl_records": 0,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
        return payload, f"warmup::{instrument}::{utc_text(start)}"

    def record_historical_warmup(
        self,
        *,
        instrument: str,
        fetched_at: datetime,
        response_sha256: str,
        request_sha256: str,
        candle: dict[str, Any],
    ) -> None:
        """Append one verified feature-only candle without creating decisions."""
        self.record_historical_warmup_batch(
            instrument=instrument,
            fetched_at=fetched_at,
            response_sha256=response_sha256,
            request_sha256=request_sha256,
            candles=[candle],
        )

    def record_historical_warmup_batch(
        self,
        *,
        instrument: str,
        fetched_at: datetime,
        response_sha256: str,
        request_sha256: str,
        candles: list[dict[str, Any]],
    ) -> None:
        """Validate a response fully, then append its new rows in one ledger write."""
        normalized = [
            self._historical_warmup_payload(
                instrument=instrument,
                fetched_at=fetched_at,
                response_sha256=response_sha256,
                request_sha256=request_sha256,
                candle=candle,
            )
            for candle in candles
        ]
        planned: list[dict[str, Any]] = []
        ledger = self.ledgers["historical_warmup_m5"]
        for payload, record_id in normalized:
            existing = ledger.by_id.get(record_id)
            if existing is not None:
                if existing["payload"].get("input_sha256") != payload["input_sha256"]:
                    raise IntegrityError("HISTORICAL_WARMUP_CONFLICT")
                continue
            ledger.plan(payload, record_id, planned)
        ledger.append_rows(planned)
        self._persist(copy.deepcopy(self.state))

    def connect_established(self, arrival: datetime) -> None:
        state = copy.deepcopy(self.state)
        state.update(
            run_state="RUNNING",
            feed_connected=True,
            connection_established=True,
            segment_started_at_utc=utc_text(arrival),
            last_arrival_utc=utc_text(arrival),
        )
        state["counters"]["connections"] += 1
        self._append(
            "control",
            {
                "event": "LIVE_PRICING_CONNECTED",
                "host": STREAM_HOST,
                "symbols": list(SYMBOLS),
                "segment_id": state["segment_id"],
                "segment_started_at_utc": state["segment_started_at_utc"],
                "feed_service_attestation_hash": state["feed_service_attestation_hash"],
                "feed_provenance_status": state["feed_provenance_status"],
                "at_utc": utc_text(arrival),
            },
            f"connect::{state['counters']['connections']}",
        )
        self._persist(state)

    def invalidate(self, reason: str, arrival: datetime | None = None) -> None:
        state = copy.deepcopy(self.state)
        if state["feed_blocked"]:
            return
        state.update(run_state="HALTED_QUALITY", feed_connected=False, feed_blocked=True, block_reason=reason)
        if reason in {"HEARTBEAT_GAP", "LOCAL_ARRIVAL_GAP"}:
            state["counters"]["gaps"] += 1
        elif reason in {"SOURCE_TIME_REGRESSION", "LOCAL_CLOCK_REVERSAL"}:
            state["counters"]["out_of_order"] += 1
        elif reason == "MALFORMED_OR_UNKNOWN_STREAM_OBJECT":
            state["counters"]["malformed"] += 1
        self._append(
            "control",
            {
                "event": "FEED_INVALID",
                "reason": reason,
                "segment_id": state.get("segment_id"),
                "segment_started_at_utc": state.get("segment_started_at_utc"),
                "feed_service_attestation_hash": state.get("feed_service_attestation_hash"),
                "feed_provenance_status": state.get("feed_provenance_status"),
                "at_utc": utc_text(arrival or datetime.now(timezone.utc)),
                "external_orders": 0,
            },
            f"halt::{reason}::{state.get('segment_id') or LEGACY_SEGMENT_ID}",
        )
        self._persist(state)

    def _validate_arrival(self, state: dict[str, Any], arrival: datetime) -> None:
        prior_text = state.get("last_arrival_utc")
        if prior_text:
            delta = (arrival - parse_utc(prior_text)).total_seconds()
            if delta < 0:
                raise FeedQualityError("LOCAL_CLOCK_REVERSAL")
            if delta > MAX_HEARTBEAT_GAP_SECONDS:
                raise FeedQualityError("LOCAL_ARRIVAL_GAP")
        state["last_arrival_utc"] = utc_text(arrival)

    def ingest_line(self, raw_line: bytes, arrival: datetime) -> dict[str, Any] | None:
        if self.state["feed_blocked"]:
            return None
        raw_hash = hashlib.sha256(raw_line).hexdigest()
        try:
            payload = json.loads(raw_line.decode("utf-8", "strict"))
            if not isinstance(payload, dict):
                raise ValueError
            state = copy.deepcopy(self.state)
            self._validate_arrival(state, arrival)
            if not isinstance(state.get("segment_id"), str) or not state["segment_id"]:
                raise FeedQualityError("SEGMENT_IDENTITY_MISSING")
            if not isinstance(state.get("segment_started_at_utc"), str):
                raise FeedQualityError("SEGMENT_IDENTITY_MISSING")
            provenance_status = state.get("feed_provenance_status")
            feed_attestation = state.get("feed_service_attestation_hash")
            if provenance_status == "ATTESTED":
                if not valid_sha256(feed_attestation):
                    raise FeedQualityError("FEED_SERVICE_ATTESTATION_INVALID")
            elif provenance_status != "LEGACY_MIGRATION_UNATTESTED" or feed_attestation is not None:
                raise FeedQualityError("FEED_SERVICE_ATTESTATION_INVALID")
            object_type = payload.get("type")
            if object_type == "HEARTBEAT":
                source = parse_oanda_time(str(payload["time"]))
                prior = state["last_source_time"].get("HEARTBEAT")
                if prior and source < parse_oanda_time(prior):
                    raise FeedQualityError("SOURCE_TIME_REGRESSION")
                state["last_source_time"]["HEARTBEAT"] = utc_text(source)
                heartbeat = {
                    "type": "HEARTBEAT",
                    "source_time_utc": utc_text(source),
                    "arrival_time_utc": utc_text(arrival),
                    "segment_id": state["segment_id"],
                    "segment_started_at_utc": state["segment_started_at_utc"],
                    "feed_service_attestation_hash": feed_attestation,
                    "feed_provenance_status": provenance_status,
                    "raw_sha256": raw_hash,
                }
                record_id = f"heartbeat::{state['segment_id']}::{raw_hash}"
                existing = self.ledgers["feed_quality"].by_id.get(record_id)
                if existing is not None:
                    existing_fact = {
                        key: value for key, value in existing["payload"].items()
                        if key != "arrival_time_utc"
                    }
                    current_fact = {
                        key: value for key, value in heartbeat.items()
                        if key != "arrival_time_utc"
                    }
                    if existing_fact != current_fact:
                        raise FeedQualityError("CONFLICTING_DUPLICATE")
                    state["counters"].setdefault("duplicate_heartbeats", 0)
                    state["counters"]["duplicate_heartbeats"] += 1
                    self._persist(state)
                    return {**heartbeat, "duplicate": True}
                state["counters"]["heartbeats"] += 1
                state["segment_heartbeats"] = int(state.get("segment_heartbeats", 0)) + 1
                self._append("feed_quality", heartbeat, record_id)
                self._persist(state)
                return heartbeat
            instrument = payload.get("instrument")
            if instrument not in SYMBOLS or object_type not in {None, "PRICE"}:
                raise ValueError
            bids, asks = payload.get("bids"), payload.get("asks")
            if not isinstance(bids, list) or not bids or not isinstance(asks, list) or not asks:
                raise ValueError
            bid_levels = [
                (float(level["price"]), int(level["liquidity"]))
                for level in bids
                if isinstance(level, dict)
            ]
            ask_levels = [
                (float(level["price"]), int(level["liquidity"]))
                for level in asks
                if isinstance(level, dict)
            ]
            if (
                len(bid_levels) != len(bids)
                or len(ask_levels) != len(asks)
                or any(not math.isfinite(price) or price <= 0 or liquidity <= 0 for price, liquidity in bid_levels + ask_levels)
            ):
                raise ValueError
            bid = max(price for price, _ in bid_levels)
            ask = min(price for price, _ in ask_levels)
            bid_liquidity = sum(liquidity for price, liquidity in bid_levels if price == bid)
            ask_liquidity = sum(liquidity for price, liquidity in ask_levels if price == ask)
            if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= bid:
                raise ValueError
            source = parse_oanda_time(str(payload["time"]))
            prior = state["last_source_time"].get(instrument)
            if prior and source < parse_oanda_time(prior):
                raise FeedQualityError("SOURCE_TIME_REGRESSION")
            event = {
                "event_id": canonical_hash({"provider": PROVIDER, "instrument": instrument, "time": utc_text(source), "bid": bid, "ask": ask}),
                "feed_identity": PROVIDER,
                "instrument": instrument,
                "event_time_utc": utc_text(source),
                "arrival_time_utc": utc_text(arrival),
                "bid": bid,
                "ask": ask,
                "bid_liquidity": bid_liquidity,
                "ask_liquidity": ask_liquidity,
                "spread": ask - bid,
                "tradeable": payload.get("status") == "tradeable",
                "continuity": CONTINUITY,
                "lossless": LOSSLESS,
                "segment_id": state["segment_id"],
                "segment_started_at_utc": state["segment_started_at_utc"],
                "feed_service_attestation_hash": feed_attestation,
                "feed_provenance_status": provenance_status,
                "raw_sha256": raw_hash,
            }
            event_digest = _market_event_digest(event)
            raw_record_id = f"event::{event['event_id']}"
            existing = self.ledgers["raw_bbo"].by_id.get(raw_record_id)
            if existing is not None:
                existing_digest = _market_event_digest(existing["payload"])
                if existing_digest != event_digest:
                    raise FeedQualityError("CONFLICTING_DUPLICATE")
                state["counters"]["duplicate_events"] += 1
                self._persist(state)
                return None
            state["last_source_time"][instrument] = utc_text(source)
            fresh_symbols = set(state.get("fresh_symbols", []))
            fresh_symbols.add(instrument)
            state["fresh_symbols"] = sorted(fresh_symbols)
            state["counters"]["market_events_received"] += 1
            state["counters"]["market_events_accepted"] += 1
            raw_row = self._append("raw_bbo", event, raw_record_id)
            self._append(
                "feed_quality",
                {
                    "accepted": True,
                    "event_id": event["event_id"],
                    "raw_record_hash": raw_row["record_hash"],
                    "segment_id": state["segment_id"],
                    "segment_started_at_utc": state["segment_started_at_utc"],
                    "feed_service_attestation_hash": feed_attestation,
                    "feed_provenance_status": provenance_status,
                },
                f"quality::{event['event_id']}",
            )
            self._persist(state)
            return event
        except FeedQualityError as exc:
            self.invalidate(str(exc), arrival)
        except Exception:
            self.invalidate("MALFORMED_OR_UNKNOWN_STREAM_OBJECT", arrival)
        return None

    def finish(self) -> None:
        state = copy.deepcopy(self.state)
        if not state["feed_blocked"]:
            state.update(run_state="STOPPED_GRACEFULLY", feed_connected=False)
        self._append(
            "pnl",
            {"terminal_mtm_included": True, "realized_return": 0.0, "unrealized_return": 0.0, "open_inventory_count": 0},
            "pnl::terminal",
        )
        state["counters"]["pnl_records"] = 1
        self._persist(state)

    def status(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "host": STREAM_HOST,
            "symbols": list(SYMBOLS),
            "continuity": CONTINUITY,
            "lossless": LOSSLESS,
            "run_state": self.state["run_state"],
            "feed_connected": self.state["feed_connected"],
            "connection_established": self.state["connection_established"],
            "segment_id": self.state.get("segment_id"),
            "segment_started_at_utc": self.state.get("segment_started_at_utc"),
            "feed_service_attestation_hash": self.state.get("feed_service_attestation_hash"),
            "feed_provenance_status": self.state.get("feed_provenance_status"),
            "fresh_symbol_count": len(self.state.get("fresh_symbols", [])),
            "segment_warmed": set(self.state.get("fresh_symbols", [])) == set(SYMBOLS),
            "feed_blocked": self.state["feed_blocked"],
            "block_reason": self.state["block_reason"],
            "heartbeat_current": int(self.state.get("segment_heartbeats", 0)) >= 1,
            "credential_values_absent": True,
            "live_order_authority": False,
            "external_orders": 0,
            "counters": copy.deepcopy(self.state["counters"]),
        }

    def request_stop(self, *_: object) -> None:
        self.stop_requested = True

    def run_live(
        self,
        account_id: str,
        token: str,
        max_seconds: float,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
        stop_when: Callable[[], bool] | None = None,
        runtime_hash: str | None = None,
    ) -> dict[str, Any]:
        if runtime_hash is None:
            raise IntegrityError("FEED_SERVICE_ATTESTATION_REQUIRED")
        if not valid_sha256(runtime_hash):
            raise IntegrityError("FEED_SERVICE_ATTESTATION_INVALID")
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        deadline = time.monotonic() + max_seconds
        strategy_hash = runtime_hash
        with RuntimeLock(self.runtime_root, strategy_hash) as lock:
            self.connect_started(strategy_hash)
            connection = http.client.HTTPSConnection(
                STREAM_NETLOC,
                timeout=MAX_HEARTBEAT_GAP_SECONDS,
                context=ssl.create_default_context(),
            )
            query = urllib.parse.urlencode({"instruments": ",".join(SYMBOLS), "snapshot": "true"})
            path = f"/v3/accounts/{urllib.parse.quote(account_id, safe='')}/pricing/stream?{query}"
            try:
                connection.request("GET", path, headers={"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"})
                response = connection.getresponse()
                if response.status != 200:
                    self.invalidate(f"HTTP_STATUS_{response.status}")
                    return self.status()
                connected_at = datetime.now(timezone.utc)
                self.connect_established(connected_at)
                lock.heartbeat(self.status(), connected_at)
                while not self.stop_requested and time.monotonic() < deadline:
                    line = response.readline()
                    if not line:
                        self.invalidate("STREAM_EOF")
                        break
                    arrival = datetime.now(timezone.utc)
                    event = self.ingest_line(line, arrival)
                    if event is not None and on_stream_event is not None:
                        on_stream_event(event)
                    lock.heartbeat(self.status(), arrival)
                    if self.state["feed_blocked"] or (stop_when is not None and stop_when()):
                        break
                self.finish()
                return self.status()
            except Exception:
                self.invalidate("NETWORK_OR_STREAM_FAILURE")
                return self.status()
            finally:
                connection.close()


def fetch_completed_m5_warmup(
    account_id: str,
    token: str,
    instrument: str,
    count: int,
    recorder: OandaLiveRecorder,
    *,
    connection_factory: Callable[..., Any] = http.client.HTTPSConnection,
    now_factory: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    """Fetch and persist one strict completed OANDA BID/ASK M5 prefix."""
    if instrument not in SYMBOLS or type(count) is not int or not 1 <= count <= 5000:
        raise ValueError("HISTORICAL_WARMUP_REQUEST_INVALID")
    if recorder.historical_warmup_ready(instrument, count):
        return 0
    requested_at = now_factory().astimezone(timezone.utc)
    boundary_epoch = int(requested_at.timestamp()) // 300 * 300
    to_time = datetime.fromtimestamp(boundary_epoch, timezone.utc)
    request_identity = {
        "method": "GET",
        "host": REST_HOST,
        "path_template": "/v3/accounts/{accountID}/instruments/{instrument}/candles",
        "instrument": instrument,
        "price": "BA",
        "granularity": "M5",
        "count": count,
        "to": utc_text(to_time),
    }
    request_sha256 = canonical_hash(request_identity)
    query = urllib.parse.urlencode({
        "price": "BA",
        "granularity": "M5",
        "count": count,
        "to": utc_text(to_time),
    })
    path = (
        f"/v3/accounts/{urllib.parse.quote(account_id, safe='')}/instruments/"
        f"{instrument}/candles?{query}"
    )
    recorder.mark_network_attempt()
    connection = connection_factory(
        REST_NETLOC,
        timeout=MAX_HEARTBEAT_GAP_SECONDS,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept-Datetime-Format": "RFC3339",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise FeedQualityError(f"HISTORICAL_HTTP_STATUS_{response.status}")
        raw = response.read()
    finally:
        connection.close()
    response_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8", "strict"))
    except Exception as exc:
        raise FeedQualityError("HISTORICAL_WARMUP_RESPONSE_INVALID") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("instrument") != instrument
        or payload.get("granularity") != "M5"
        or not isinstance(payload.get("candles"), list)
        or not payload["candles"]
    ):
        raise FeedQualityError("HISTORICAL_WARMUP_SCHEMA_MISMATCH")
    candles = payload["candles"]
    starts: list[datetime] = []
    for candle in candles:
        if not isinstance(candle, dict) or candle.get("complete") is not True:
            raise FeedQualityError("HISTORICAL_WARMUP_INCOMPLETE")
        try:
            start = parse_oanda_time(str(candle["time"]))
        except Exception as exc:
            raise FeedQualityError("HISTORICAL_WARMUP_TIME_INVALID") from exc
        if start + timedelta(minutes=5) > to_time or start + timedelta(minutes=5) > requested_at:
            raise FeedQualityError("HISTORICAL_WARMUP_FUTURE")
        starts.append(start)
    if starts != sorted(starts) or len(starts) != len(set(starts)):
        raise FeedQualityError("HISTORICAL_WARMUP_OVERLAP")
    if any(current - prior != timedelta(minutes=5) for prior, current in zip(starts, starts[1:])):
        raise FeedQualityError("HISTORICAL_WARMUP_GAP")
    if len(candles) != count:
        raise FeedQualityError("HISTORICAL_WARMUP_COUNT_MISMATCH")
    recorder.record_historical_warmup_batch(
        instrument=instrument,
        fetched_at=requested_at,
        response_sha256=response_sha256,
        request_sha256=request_sha256,
        candles=candles,
    )
    return len(candles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.parse_args(argv)
    print(json.dumps({
        "error": "FEED_SERVICE_ATTESTATION_REQUIRED",
        "operator_action": "use oanda_launchd_runtime.py feed",
        "network_attempts": 0,
        "credential_reads": 0,
        "external_orders": 0,
    }, sort_keys=True), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
