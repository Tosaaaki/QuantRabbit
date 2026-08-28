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
from datetime import datetime, timezone
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
STREAM_NETLOC = "stream-fxtrade.oanda.com"
APPROVED_ENV_FILE = Path("/Users/tossaki/App/QuantRabbit-live/.env.local")
SYMBOLS = ("EUR_USD", "USD_JPY")
CONTINUITY = "HEARTBEAT_ONLY"
LOSSLESS = False
MAX_HEARTBEAT_GAP_SECONDS = 15.0
LEDGERS = ("raw_bbo", "feed_quality", "decisions", "virtual_fills", "pnl", "control")


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
        if not self.contract["bot_only"] or self.contract["actual_llm_enabled"]:
            raise IntegrityError("arm boundary mismatch")

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

    def connect_started(self) -> None:
        state = copy.deepcopy(self.state)
        state["counters"].setdefault("segments", 0)
        state["counters"]["segments"] += 1
        state.update(
            run_state="CONNECTING",
            feed_connected=False,
            feed_blocked=False,
            block_reason=None,
            segment_id=f"segment-{state['counters']['segments']:08d}",
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

    def connect_established(self, arrival: datetime) -> None:
        state = copy.deepcopy(self.state)
        state.update(
            run_state="RUNNING",
            feed_connected=True,
            connection_established=True,
            last_arrival_utc=utc_text(arrival),
        )
        state["counters"]["connections"] += 1
        self._append(
            "control",
            {"event": "LIVE_PRICING_CONNECTED", "host": STREAM_HOST, "symbols": list(SYMBOLS), "at_utc": utc_text(arrival)},
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
            {"event": "FEED_INVALID", "reason": reason, "at_utc": utc_text(arrival or datetime.now(timezone.utc)), "external_orders": 0},
            f"halt::{reason}",
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
            object_type = payload.get("type")
            if object_type == "HEARTBEAT":
                source = parse_oanda_time(str(payload["time"]))
                prior = state["last_source_time"].get("HEARTBEAT")
                if prior and source < parse_oanda_time(prior):
                    raise FeedQualityError("SOURCE_TIME_REGRESSION")
                state["last_source_time"]["HEARTBEAT"] = utc_text(source)
                state["counters"]["heartbeats"] += 1
                self._append(
                    "feed_quality",
                    {"type": "HEARTBEAT", "source_time_utc": utc_text(source), "arrival_time_utc": utc_text(arrival), "raw_sha256": raw_hash},
                    f"heartbeat::{raw_hash}",
                )
                self._persist(state)
                return {
                    "type": "HEARTBEAT",
                    "source_time_utc": utc_text(source),
                    "arrival_time_utc": utc_text(arrival),
                }
            instrument = payload.get("instrument")
            if instrument not in SYMBOLS or object_type not in {None, "PRICE"}:
                raise ValueError
            bids, asks = payload.get("bids"), payload.get("asks")
            if not isinstance(bids, list) or not bids or not isinstance(asks, list) or not asks:
                raise ValueError
            bid = max(float(level["price"]) for level in bids if isinstance(level, dict))
            ask = min(float(level["price"]) for level in asks if isinstance(level, dict))
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
                "spread": ask - bid,
                "tradeable": payload.get("status") == "tradeable",
                "continuity": CONTINUITY,
                "lossless": LOSSLESS,
                "raw_sha256": raw_hash,
            }
            event_digest = canonical_hash(
                {key: value for key, value in event.items() if key != "arrival_time_utc"}
            )
            raw_record_id = f"event::{event['event_id']}"
            existing = self.ledgers["raw_bbo"].by_id.get(raw_record_id)
            if existing is not None:
                existing_digest = canonical_hash(
                    {key: value for key, value in existing["payload"].items() if key != "arrival_time_utc"}
                )
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
                {"accepted": True, "event_id": event["event_id"], "raw_record_hash": raw_row["record_hash"]},
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
            "fresh_symbol_count": len(self.state.get("fresh_symbols", [])),
            "segment_warmed": set(self.state.get("fresh_symbols", [])) == set(SYMBOLS),
            "feed_blocked": self.state["feed_blocked"],
            "block_reason": self.state["block_reason"],
            "heartbeat_current": self.state["counters"]["heartbeats"] >= 1,
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
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        deadline = time.monotonic() + max_seconds
        strategy_hash = runtime_hash or canonical_hash({"contract": self.contract_hash, "provider": PROVIDER})
        with RuntimeLock(self.runtime_root, strategy_hash) as lock:
            self.connect_started()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        account_id, token = load_approved_live_credentials()
        recorder = OandaLiveRecorder(args.runtime_root)
        recorder.mark_approved_credential_file_read()
        status = recorder.run_live(account_id, token, args.seconds)
        print(json.dumps(status, sort_keys=True))
        return 2 if status["feed_blocked"] else 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
