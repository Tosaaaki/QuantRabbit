#!/usr/bin/env python3
"""Acquire a bounded, read-only OANDA truth packet for operator-alpha research.

This script performs GET requests only.  It never opens, changes, or closes an
order and never writes to the live checkout.  Output is a sanitized subset of
transactions and complete bid/ask candles under this research directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.broker.oanda import OandaReadOnlyClient


ROOT = Path(__file__).resolve().parent
TRANSACTIONS_OUT = ROOT / "source_transactions_v1.json"
CANDLES_OUT = ROOT / "source_candles_v1.json"
MANIFEST_OUT = ROOT / "source_manifest_v1.json"

FROZEN_TRADES = (
    {"entry_fill_id": "473162", "close_fill_id": "473180", "label": "margin_closeout_1"},
    {"entry_fill_id": "473183", "close_fill_id": "473186", "label": "margin_closeout_2"},
    {"entry_fill_id": "473189", "close_fill_id": "473191", "label": "manual_win_1"},
    {"entry_fill_id": "473193", "close_fill_id": "473195", "label": "manual_win_2"},
    {"entry_fill_id": "473197", "close_fill_id": "473199", "label": "manual_win_3"},
    {"entry_fill_id": "473201", "close_fill_id": "473204", "label": "manual_win_4"},
)
BOUNDARY_TRANSACTION_IDS = frozenset({"473202", "473205", "473207", "473208", "473209"})
ALLOWED_TOP_LEVEL = (
    "id",
    "time",
    "type",
    "reason",
    "instrument",
    "units",
    "price",
    "pl",
    "financing",
    "commission",
    "accountBalance",
    "orderID",
    "tradeID",
    "tradeOpened",
    "tradesClosed",
)
ALLOWED_TRADE_FIELDS = (
    "price",
    "tradeID",
    "units",
    "realizedPL",
    "financing",
    "halfSpreadCost",
    "initialMarginRequired",
    "homeConversionCost",
    "plHomeConversionCost",
)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _oanda_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_trade(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ALLOWED_TRADE_FIELDS if key in value}


def _sanitize_transaction(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ALLOWED_TOP_LEVEL:
        if key not in value:
            continue
        item = value[key]
        if key == "tradeOpened" and isinstance(item, dict):
            result[key] = _sanitize_trade(item)
        elif key == "tradesClosed" and isinstance(item, list):
            result[key] = [_sanitize_trade(row) for row in item if isinstance(row, dict)]
        else:
            result[key] = item
    return result


def _candle_row(value: dict[str, Any]) -> dict[str, Any] | None:
    if value.get("complete") is not True:
        return None
    bid = value.get("bid")
    ask = value.get("ask")
    if not isinstance(bid, dict) or not isinstance(ask, dict):
        return None
    return {
        "time": value.get("time"),
        "volume": int(value.get("volume") or 0),
        "complete": True,
        "bid": {key: str(bid[key]) for key in ("o", "h", "l", "c")},
        "ask": {key: str(ask[key]) for key in ("o", "h", "l", "c")},
    }


def _fetch_candles(
    client: OandaReadOnlyClient,
    *,
    pair: str,
    granularity: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {
            "granularity": granularity,
            "from": _oanda_time(start),
            "to": _oanda_time(end),
            "price": "BA",
            "includeFirst": "true",
        },
    )
    return [row for item in payload.get("candles", []) if (row := _candle_row(item)) is not None]


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(*, env_file: Path) -> dict[str, Any]:
    client = OandaReadOnlyClient(env_file=env_file)
    payload = client.transactions_since_id("473160")
    by_id = {str(tx.get("id")): tx for tx in payload.get("transactions", []) if isinstance(tx, dict)}
    required_ids = {
        value
        for trade in FROZEN_TRADES
        for value in (trade["entry_fill_id"], trade["close_fill_id"])
    }
    missing = sorted(required_ids - by_id.keys(), key=int)
    if missing:
        raise RuntimeError(f"missing frozen OANDA transaction ids: {missing}")

    referenced_order_ids = {
        str(by_id[value].get("orderID"))
        for value in required_ids | BOUNDARY_TRANSACTION_IDS
        if value in by_id and by_id[value].get("orderID")
    }
    included_ids = required_ids | referenced_order_ids | BOUNDARY_TRANSACTION_IDS
    transactions = [_sanitize_transaction(by_id[value]) for value in sorted(included_ids & by_id.keys(), key=int)]
    transaction_packet = {
        "contract": "OPERATOR_ALPHA_TRANSACTION_TRUTH_V1",
        "read_only": True,
        "since_transaction_id": "473160",
        "broker_last_transaction_id_at_acquisition": str(payload.get("lastTransactionID") or ""),
        "frozen_trades": list(FROZEN_TRADES),
        "transactions": transactions,
    }

    candle_packets: list[dict[str, Any]] = []
    for trade in FROZEN_TRADES:
        entry = by_id[trade["entry_fill_id"]]
        entry_time = _parse_time(str(entry["time"]))
        pair = str(entry["instrument"])
        # Keep the same forty-minute counterfactual observation window for
        # winners and losses. Truncating a winner at its actual close would
        # make a later operator timeout look observed when the path was merely
        # missing.
        path_end = entry_time + timedelta(minutes=40)
        windows = (
            ("S5", entry_time - timedelta(minutes=1), path_end),
            ("M5", entry_time - timedelta(hours=4), path_end),
            ("H4", entry_time - timedelta(days=14), entry_time),
        )
        for granularity, start, end in windows:
            rows = _fetch_candles(
                client,
                pair=pair,
                granularity=granularity,
                start=start,
                end=end,
            )
            if not rows:
                raise RuntimeError(f"no complete {granularity} bid/ask rows for {trade['entry_fill_id']}")
            candle_packets.append(
                {
                    "entry_fill_id": trade["entry_fill_id"],
                    "pair": pair,
                    "granularity": granularity,
                    "requested_from_utc": _oanda_time(start),
                    "requested_to_utc": _oanda_time(end),
                    "rows": rows,
                }
            )

    candle_packet = {
        "contract": "OPERATOR_ALPHA_CANDLE_TRUTH_V1",
        "read_only": True,
        "price_component": "BID_ASK",
        "complete_only": True,
        "packets": candle_packets,
    }
    _write_json(TRANSACTIONS_OUT, transaction_packet)
    _write_json(CANDLES_OUT, candle_packet)
    manifest = {
        "contract": "OPERATOR_ALPHA_SOURCE_MANIFEST_V1",
        "permissions": {
            "live": False,
            "paper": False,
            "broker_mutation": False,
            "orders": False,
            "deploy": False,
            "broker_get_only": True,
        },
        "files": {
            TRANSACTIONS_OUT.name: {"sha256": _sha(TRANSACTIONS_OUT), "bytes": TRANSACTIONS_OUT.stat().st_size},
            CANDLES_OUT.name: {"sha256": _sha(CANDLES_OUT), "bytes": CANDLES_OUT.stat().st_size},
        },
        "canonical_payload_sha256": {
            "transactions": hashlib.sha256(_canonical_bytes(transaction_packet)).hexdigest(),
            "candles": hashlib.sha256(_canonical_bytes(candle_packet)).hexdigest(),
        },
    }
    _write_json(MANIFEST_OUT, manifest)
    return manifest


def verify_existing() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = ROOT / name
        actual = _sha(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"source hash mismatch for {name}: {actual}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = verify_existing()
    else:
        if args.env_file is None:
            parser.error("--env-file is required for read-only acquisition")
        result = acquire(env_file=args.env_file)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
