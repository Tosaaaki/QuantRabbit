"""OANDA LIVE Step 3/4: causal M5 observation and zero-order virtual accounting."""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
import ssl
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oanda_live_feed import (
    REST_HOST,
    SYMBOLS,
    OandaLiveRecorder,
    load_approved_live_credentials,
    parse_oanda_time,
)
from shadow_runtime import HashLedger, canonical_hash, real_dir, utc_text

REST_NETLOC = "api-fxtrade.oanda.com"
R5_COMMIT = "15c7d205a78a14116651046b5fa2741d37e72cf2"
R5_COMBINED_SHA256 = "a6f2c4990af89462d33dca0bee8fca0c3af700562de4f074c5f90331d3d87d13"
R5_CANDIDATE_SCOPE = "ACCOUNTING_ONLY_NOT_CAUSAL_SIGNAL_ADMISSION"
CANARY_ARM = "PLUMBING_CANARY_NON_EVIDENCE"
STEP_LEDGERS = (
    "historical_warmup_m5",
    "forward_completed_m5",
    "r5_proposals",
    "canary_non_evidence",
    "decisions",
    "virtual_fills",
    "inventory",
    "pnl",
    "control",
)
EXECUTION_ARMS = {
    "EXECUTABLE_BASE": {
        "commission_ppm_per_side": 10,
        "financing_ppm_per_day": 5,
        "slippage_pips_per_side": 0.0,
    },
    "ADVERSE_STRESS": {
        "commission_ppm_per_side": 20,
        "financing_ppm_per_day": 10,
        "slippage_pips_per_side": 1.0,
    },
}


def candle_content_hash(candle: dict[str, Any]) -> str:
    return canonical_hash(candle)


class Step34Runtime:
    def __init__(self, runtime_root: Path):
        self.runtime_root = Path(runtime_root)
        self.ledger_root = self.runtime_root / "step34_ledgers"
        real_dir(self.runtime_root)
        real_dir(self.ledger_root)
        self.ledgers = {
            name: HashLedger(self.ledger_root / f"{name}.jsonl") for name in STEP_LEDGERS
        }
        self.builders: dict[str, dict[str, Any]] = {}
        self.forward_bar_symbols: set[str] = {
            row["payload"]["instrument"]
            for row in self.ledgers["forward_completed_m5"].rows
            if "instrument" in row["payload"]
        }
        self.canary_complete = bool(self.ledgers["virtual_fills"].rows)
        self.pending_canary: dict[str, Any] | None = self._recover_pending_canary()

    def _recover_pending_canary(self) -> dict[str, Any] | None:
        if self.canary_complete or not self.ledgers["canary_non_evidence"].rows:
            return None
        proposal_row = self.ledgers["canary_non_evidence"].rows[-1]
        proposal = proposal_row["payload"]
        for decision_row in reversed(self.ledgers["decisions"].rows):
            if decision_row["payload"].get("proposal_id") == proposal.get("proposal_id"):
                return {
                    **proposal,
                    "proposal_record_hash": proposal_row["record_hash"],
                    "decision_record_hash": decision_row["record_hash"],
                }
        return None

    def append(self, ledger: str, payload: dict[str, Any], record_id: str) -> dict[str, Any]:
        planned: list[dict[str, Any]] = []
        row = self.ledgers[ledger].plan(payload, record_id, planned)
        self.ledgers[ledger].append_rows(planned)
        return row

    def record_warmup(self, instrument: str, fetched_at: datetime, candle: dict[str, Any]) -> None:
        if candle.get("complete") is not True:
            return
        source = parse_oanda_time(str(candle["time"]))
        digest = candle_content_hash(candle)
        record_id = f"warmup::{instrument}::{utc_text(source)}::{digest}"
        if any(row["record_id"] == record_id for row in self.ledgers["historical_warmup_m5"].rows):
            return
        self.append(
            "historical_warmup_m5",
            {
                "instrument": instrument,
                "timeframe": "M5",
                "fetched_at_utc": utc_text(fetched_at),
                "source_candle_time_utc": utc_text(source),
                "complete": True,
                "content_sha256": digest,
                "excluded_from_forward_pnl": True,
                "bid": candle["bid"],
                "ask": candle["ask"],
            },
            record_id,
        )

    @staticmethod
    def _bucket_start(stamp: datetime) -> datetime:
        epoch = int(stamp.timestamp()) // 300 * 300
        return datetime.fromtimestamp(epoch, timezone.utc)

    def _new_builder(self, event: dict[str, Any], start: datetime) -> dict[str, Any]:
        bid, ask = float(event["bid"]), float(event["ask"])
        return {
            "instrument": event["instrument"],
            "timeframe": "M5",
            "bucket_start_utc": utc_text(start),
            "bucket_end_utc": utc_text(start + timedelta(minutes=5)),
            "first_source_time_utc": event["event_time_utc"],
            "last_source_time_utc": event["event_time_utc"],
            "first_arrival_time_utc": event["arrival_time_utc"],
            "arrival_watermark_utc": event["arrival_time_utc"],
            "event_count": 1,
            "bid_o": bid,
            "bid_h": bid,
            "bid_l": bid,
            "bid_c": bid,
            "ask_o": ask,
            "ask_h": ask,
            "ask_l": ask,
            "ask_c": ask,
            "completed_from_live_price_events": True,
            "historical_warmup_used_for_pnl": False,
        }

    def _close_bar(self, builder: dict[str, Any]) -> dict[str, Any]:
        payload = {**builder, "complete": True, "source_arrival_chronology_frozen": True}
        row = self.append(
            "forward_completed_m5",
            payload,
            f"forward-m5::{payload['instrument']}::{payload['bucket_start_utc']}",
        )
        self.forward_bar_symbols.add(payload["instrument"])
        return {**payload, "bar_record_hash": row["record_hash"]}

    def _update_builder(self, builder: dict[str, Any], event: dict[str, Any]) -> None:
        bid, ask = float(event["bid"]), float(event["ask"])
        builder["event_count"] += 1
        builder["last_source_time_utc"] = event["event_time_utc"]
        builder["arrival_watermark_utc"] = event["arrival_time_utc"]
        builder["bid_h"] = max(builder["bid_h"], bid)
        builder["bid_l"] = min(builder["bid_l"], bid)
        builder["bid_c"] = bid
        builder["ask_h"] = max(builder["ask_h"], ask)
        builder["ask_l"] = min(builder["ask_l"], ask)
        builder["ask_c"] = ask

    def on_stream_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "HEARTBEAT":
            return
        self._maybe_fill_canary(event)
        instrument = str(event["instrument"])
        source = parse_oanda_time(str(event["event_time_utc"]))
        start = self._bucket_start(source)
        builder = self.builders.get(instrument)
        if builder is None:
            self.builders[instrument] = self._new_builder(event, start)
            return
        prior_start = parse_oanda_time(builder["bucket_start_utc"])
        if start < prior_start:
            raise RuntimeError("FORWARD_M5_SOURCE_REGRESSION")
        if start == prior_start:
            self._update_builder(builder, event)
            return
        closed = self._close_bar(builder)
        self.builders[instrument] = self._new_builder(event, start)
        self._evaluate_r5(closed, event)

    def _evaluate_r5(self, closed_bar: dict[str, Any], event: dict[str, Any]) -> None:
        self.append(
            "control",
            {
                "event": "R5_NATURAL_PROPOSAL_NOT_EMITTED",
                "candidate_scope": R5_CANDIDATE_SCOPE,
                "reason": "FROZEN_R5_HAS_NO_ADMITTED_CAUSAL_SIGNAL_INTERFACE",
                "bar_record_hash": closed_bar["bar_record_hash"],
                "profit_evidence": False,
            },
            f"r5-no-proposal::{closed_bar['bar_record_hash']}",
        )
        if self.pending_canary is None and not self.canary_complete and self.forward_bar_symbols == set(SYMBOLS):
            proposal_id = f"canary::{closed_bar['bar_record_hash']}"
            proposal = {
                "arm": CANARY_ARM,
                "proposal_id": proposal_id,
                "instrument": "EUR_USD",
                "direction": 1,
                "decision_source_time_utc": event["event_time_utc"],
                "decision_arrival_time_utc": event["arrival_time_utc"],
                "completed_bar_hash": closed_bar["bar_record_hash"],
                "r5_commit": R5_COMMIT,
                "r5_combined_sha256": R5_COMBINED_SHA256,
                "natural_r5_signal": False,
                "profit_evidence": False,
                "r5_result_included": False,
                "adoption_evidence": False,
                "external_submission_allowed": False,
            }
            proposal_row = self.append("canary_non_evidence", proposal, f"proposal::{proposal_id}")
            decision = {
                **proposal,
                "proposal_record_hash": proposal_row["record_hash"],
                "decision": "DIAGNOSTIC_LONG_ONCE",
                "llm_called": False,
            }
            decision_row = self.append("decisions", decision, f"decision::{proposal_id}")
            self.pending_canary = {
                **proposal,
                "proposal_record_hash": proposal_row["record_hash"],
                "decision_record_hash": decision_row["record_hash"],
            }

    def _maybe_fill_canary(self, event: dict[str, Any]) -> None:
        pending = self.pending_canary
        if pending is None or self.canary_complete:
            return
        if event["instrument"] != pending["instrument"]:
            return
        if parse_oanda_time(event["arrival_time_utc"]) <= parse_oanda_time(pending["decision_arrival_time_utc"]):
            return
        bid, ask = float(event["bid"]), float(event["ask"])
        if not math.isfinite(bid) or not math.isfinite(ask):
            raise RuntimeError("CANARY_INVALID_BBO")
        pip = 0.0001
        for arm, costs in EXECUTION_ARMS.items():
            entry = ask + costs["slippage_pips_per_side"] * pip
            fill = {
                "canary_arm": CANARY_ARM,
                "execution_arm": arm,
                "proposal_id": pending["proposal_id"],
                "proposal_record_hash": pending["proposal_record_hash"],
                "decision_record_hash": pending["decision_record_hash"],
                "instrument": event["instrument"],
                "direction": 1,
                "first_post_decision_bbo_event_id": event["event_id"],
                "fill_source_time_utc": event["event_time_utc"],
                "fill_arrival_time_utc": event["arrival_time_utc"],
                "bid": bid,
                "ask": ask,
                "virtual_entry_price": entry,
                "costs": costs,
                "profit_evidence": False,
                "r5_result_included": False,
                "external_submission_allowed": False,
            }
            fill_row = self.append(
                "virtual_fills", fill, f"fill::{pending['proposal_id']}::{arm}"
            )
            inventory = {
                "canary_arm": CANARY_ARM,
                "execution_arm": arm,
                "instrument": event["instrument"],
                "direction": 1,
                "entry_price": entry,
                "fill_record_hash": fill_row["record_hash"],
                "terminal_mark_bid": bid,
                "open": True,
                "profit_evidence": False,
            }
            self.append("inventory", inventory, f"inventory::{pending['proposal_id']}::{arm}")
            gross = bid / entry - 1.0
            net = gross - 2 * costs["commission_ppm_per_side"] * 1e-6
            self.append(
                "pnl",
                {
                    "canary_arm": CANARY_ARM,
                    "execution_arm": arm,
                    "instrument": event["instrument"],
                    "terminal_mtm_included": True,
                    "gross_return": gross,
                    "net_return": net,
                    "realized_return": 0.0,
                    "unrealized_return": net,
                    "profit_evidence": False,
                    "r5_result_included": False,
                },
                f"pnl::{pending['proposal_id']}::{arm}",
            )
        self.canary_complete = True
        self.pending_canary = None

    def status(self) -> dict[str, Any]:
        return {
            "r5_candidate_scope": R5_CANDIDATE_SCOPE,
            "warmup_completed_m5": len(self.ledgers["historical_warmup_m5"].rows),
            "forward_completed_m5": len(self.ledgers["forward_completed_m5"].rows),
            "forward_bar_symbols": sorted(self.forward_bar_symbols),
            "natural_r5_proposals": len(self.ledgers["r5_proposals"].rows),
            "canary_proposals": len(self.ledgers["canary_non_evidence"].rows),
            "decisions": len(self.ledgers["decisions"].rows),
            "virtual_fills": len(self.ledgers["virtual_fills"].rows),
            "inventory_records": len(self.ledgers["inventory"].rows),
            "pnl_records": len(self.ledgers["pnl"].rows),
            "llm_calls": 0,
            "external_order_attempts": 0,
            "external_orders": 0,
            "canary_complete": self.canary_complete,
            "canary_profit_evidence": False,
            "r5_profit_unproven": True,
        }


def fetch_completed_m5(
    account_id: str,
    token: str,
    instrument: str,
    count: int,
    recorder: OandaLiveRecorder,
    runtime: Step34Runtime,
    connection_factory: Any = http.client.HTTPSConnection,
) -> int:
    if instrument not in SYMBOLS:
        raise ValueError("UNEXPECTED_INSTRUMENT")
    connection = connection_factory(REST_NETLOC, timeout=15.0, context=ssl.create_default_context())
    query = urllib.parse.urlencode({"price": "BA", "granularity": "M5", "count": count})
    quoted_account = urllib.parse.quote(account_id, safe="")
    path = f"/v3/accounts/{quoted_account}/instruments/{instrument}/candles?{query}"
    state = copy.deepcopy(recorder.state)
    state["counters"]["network_attempts"] += 1
    recorder._persist(state)
    try:
        connection.request(
            "GET",
            path,
            headers={"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HISTORICAL_HTTP_STATUS_{response.status}")
        raw = response.read()
        payload = json.loads(raw.decode("utf-8", "strict"))
        if payload.get("instrument") != instrument or payload.get("granularity") != "M5":
            raise RuntimeError("HISTORICAL_SCHEMA_MISMATCH")
        fetched_at = datetime.now(timezone.utc)
        completed = [c for c in payload.get("candles", []) if isinstance(c, dict) and c.get("complete") is True]
        for candle in completed:
            runtime.record_warmup(instrument, fetched_at, candle)
        return len(completed)
    finally:
        connection.close()


def credential_absence(runtime_root: Path, account_id: str, token: str) -> bool:
    secrets = (account_id.encode(), token.encode())
    for path in Path(runtime_root).rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if any(secret in data for secret in secrets):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=420.0)
    parser.add_argument("--warmup-count", type=int, default=300)
    args = parser.parse_args(argv)
    account_id = token = ""
    try:
        account_id, token = load_approved_live_credentials()
        recorder = OandaLiveRecorder(args.runtime_root)
        recorder.mark_approved_credential_file_read()
        runtime = Step34Runtime(args.runtime_root)
        warmup_counts = {
            instrument: fetch_completed_m5(
                account_id, token, instrument, args.warmup_count, recorder, runtime
            )
            for instrument in SYMBOLS
        }
        feed_status = recorder.run_live(
            account_id,
            token,
            args.seconds,
            on_stream_event=runtime.on_stream_event,
            stop_when=lambda: runtime.canary_complete,
        )
        result = {
            "pid": os.getpid(),
            "host": feed_status["host"],
            "symbols": feed_status["symbols"],
            "feed_connected": feed_status["feed_connected"],
            "connection_established": feed_status["connection_established"],
            "heartbeat_current": feed_status["heartbeat_current"],
            "feed_blocked": feed_status["feed_blocked"],
            "block_reason": feed_status["block_reason"],
            "warmup_fetched": warmup_counts,
            "market_events": feed_status["counters"]["market_events_accepted"],
            "heartbeats": feed_status["counters"]["heartbeats"],
            "network_attempts": feed_status["counters"]["network_attempts"],
            "credential_reads": feed_status["counters"]["credential_reads"],
            **runtime.status(),
            "credential_values_absent": credential_absence(args.runtime_root, account_id, token),
        }
        print(json.dumps(result, sort_keys=True))
        if feed_status["feed_blocked"] or not result["credential_values_absent"]:
            return 2
        if not runtime.canary_complete:
            return 4
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
