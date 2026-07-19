"""Virtual broker: OANDA-mechanics paper account priced by REAL quotes.

Operator directive (2026-07-19): a virtual environment identical to
reality where the duty agent can trade at will.  This module is the
broker core; feeds (live polling or historical replay) supply quotes.

Honesty properties:
  * Fills happen ONLY at quotes actually supplied by the feed — market
    orders fill at the current real ask/bid; limit/TP/SL fill when a
    supplied quote touches the level, at the level (or the quote when
    it gapped past, whichever is worse for the trader).  No synthesis.
  * Accounting mirrors OANDA: hedge netting (margin on the larger side
    per instrument), leverage cap, margin closeout at 100% usage
    liquidating everything at current quotes.
  * Every action and fill is written to a hash-chained append-only
    ledger with the exact quote that caused it.
  * The broker never talks to the real broker.  It cannot place real
    orders by construction.

Non-JPY-quote pairs convert P&L at the latest USD_JPY mid supplied to
the broker (declared approximation, logged per fill).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

UTC = timezone.utc

LEVERAGE_DEFAULT = 25.0
CLOSEOUT_USAGE = 1.0
MAX_CONVERSION_QUOTE_AGE_S = 90.0
SNAPSHOT_SCHEMA = "QR_VIRTUAL_BROKER_SNAPSHOT_V2"
QUOTE_BATCH_CONTRACT = "QR_VIRTUAL_QUOTE_BATCH_V1"
ACCOUNT_MARK_CONTRACT = "QR_VIRTUAL_ACCOUNT_MARK_V1"
_CHAIN_GENESIS_SHA256 = "0" * 64
_QUOTE_BATCH_RECEIPT_KEYS = {
    "contract",
    "batch_index",
    "coordinate",
    "feed_pairs",
    "batch_pairs",
    "coverage_complete",
    "quotes",
    "quotes_sha256",
    "previous_batch_sha256",
    "batch_sha256",
}
_ACCOUNT_MARK_KEYS = {
    "contract",
    "mark_index",
    "kind",
    "coordinate",
    "batch_index",
    "batch_sha256",
    "feed_cursor",
    "account",
    "account_sha256",
    "positions",
    "positions_sha256",
    "orders",
    "orders_sha256",
    "quotes",
    "quotes_sha256",
    "previous_mark_sha256",
    "mark_sha256",
}
_REPLAY_COORDINATE_KEYS = {
    "mode",
    "epoch",
    "phase",
    "granularity",
    "intrabar",
}
_SNAPSHOT_KEYS = {
    "schema",
    "balance_jpy",
    "seq",
    "positions",
    "orders",
    "quote_seq",
    "last_quotes",
    "last_quote_sequences",
    "last_quote_watermarks",
    "quote_history",
    "feed_cursor",
    "ledger_tip_sha",
}
_STATEFUL_LEDGER_EVENTS = {
    "FILL_MARKET",
    "ORDER_LIMIT",
    "ORDER_STOP",
    "ORDER_CANCEL",
    "ORDER_CANCEL_CONCURRENCY_CAP",
    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
    "FILL_LIMIT",
    "CLOSE",
    "SET_EXIT",
    "MARGIN_CLOSEOUT",
}


class VirtualBrokerError(ValueError):
    """Contract violation; callers must fail closed."""


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _round_price(pair: str, price: float) -> float:
    """Instrument price precision (JPY quotes: 3 dp, others: 5 dp) —
    mirrors broker tick precision and kills float-epsilon artifacts."""

    return round(price, 3 if pair.endswith("JPY") else 5)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise VirtualBrokerError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in pairs:
        if key in row:
            raise VirtualBrokerError(f"duplicate JSON key is forbidden: {key}")
        row[key] = value
    return row


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _canonical_json_copy(name: str, value: Any) -> Any:
    """Detach caller-owned evidence using the same JSON domain as ledger hashes."""

    _validate_finite_tree(value, name)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return _strict_json_loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VirtualBrokerError(f"{name} must be canonical JSON") from exc


def _finite_number(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise VirtualBrokerError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VirtualBrokerError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise VirtualBrokerError(f"{name} must be finite")
    if positive and number <= 0:
        raise VirtualBrokerError(f"{name} must be positive")
    if non_negative and number < 0:
        raise VirtualBrokerError(f"{name} must be non-negative")
    return number


def _validate_finite_tree(value: Any, path: str = "payload") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise VirtualBrokerError(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")
        return
    raise VirtualBrokerError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _validate_pair(pair: str) -> None:
    parts = pair.split("_")
    if len(parts) != 2 or any(
        len(part) != 3 or not part.isalpha() or not part.isupper() for part in parts
    ):
        raise VirtualBrokerError(f"invalid pair: {pair}")


@dataclass
class VBPosition:
    trade_id: str
    pair: str
    side: str  # LONG / SHORT
    units: float
    entry_price: float
    opened_ts: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


@dataclass
class VBOrder:
    order_id: str
    pair: str
    side: str
    units: float
    limit_price: float
    tp_pips: Optional[float] = None
    sl_pips: Optional[float] = None
    kind: str = "LIMIT"  # LIMIT (at level or better) / STOP (breakout trigger)


@dataclass
class VirtualBroker:
    ledger_path: Path
    balance_jpy: float = 200_000.0
    fast_ledger: bool = False  # flush without fsync (lab runs)
    slippage_pips: float = 0.0  # stress: extra pips against the trader per fill
    financing_pips_per_day: float = 0.0  # holding cost, pro-rata vs opened_ts
    leverage: float = LEVERAGE_DEFAULT
    positions: dict[str, VBPosition] = field(default_factory=dict)
    orders: dict[str, VBOrder] = field(default_factory=dict)
    last_quotes: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    _last_quote_sequences: dict[str, int] = field(default_factory=dict, repr=False)
    _last_quote_watermarks: dict[str, int] = field(default_factory=dict, repr=False)
    _quote_history: dict[str, list[tuple[float, float, str, int]]] = field(
        default_factory=dict, repr=False
    )
    _quote_seq: int = 0
    feed_cursor: Optional[dict[str, Any]] = field(default=None, repr=False)
    _entry_admission: Optional[
        Callable[[str, str, Optional[str]], Optional[dict[str, Any]]]
    ] = field(default=None, init=False, repr=False)
    _seq: int = 0
    _prev_sha: str = "0" * 64
    _batch_index: int = field(default=0, init=False, repr=False)
    _previous_batch_sha256: str = field(
        default=_CHAIN_GENESIS_SHA256, init=False, repr=False
    )
    _last_batch_receipt: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    _mark_index: int = field(default=0, init=False, repr=False)
    _previous_mark_sha256: str = field(
        default=_CHAIN_GENESIS_SHA256, init=False, repr=False
    )
    _last_phase_batch_index: int = field(default=-1, init=False, repr=False)
    _account_mark_chain_started: bool = field(default=False, init=False, repr=False)
    _mtm_terminal_emitted: bool = field(default=False, init=False, repr=False)
    _last_batch_quote_seq_before: Optional[int] = field(
        default=None, init=False, repr=False
    )
    _last_batch_applied: bool = field(default=False, init=False, repr=False)
    _continuous_evidence_resume_forbidden: bool = field(
        default=False, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.balance_jpy = _finite_number("balance_jpy", self.balance_jpy)
        self.slippage_pips = _finite_number(
            "slippage_pips", self.slippage_pips, non_negative=True
        )
        self.financing_pips_per_day = _finite_number(
            "financing_pips_per_day", self.financing_pips_per_day, non_negative=True
        )
        self.leverage = _finite_number("leverage", self.leverage, positive=True)
        requires_state_restore = False
        if self.ledger_path.exists():
            expected_prev = "0" * 64
            with self.ledger_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise VirtualBrokerError(
                            f"blank ledger record at line {line_number}"
                        )
                    try:
                        record = _strict_json_loads(line)
                    except (json.JSONDecodeError, VirtualBrokerError) as exc:
                        raise VirtualBrokerError(
                            f"invalid ledger JSON at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise VirtualBrokerError(
                            f"invalid ledger record at line {line_number}"
                        )
                    supplied_sha = record.get("sha")
                    body = {key: value for key, value in record.items() if key != "sha"}
                    if set(body) != {"ts_utc", "event", "payload", "prev_sha"}:
                        raise VirtualBrokerError(
                            f"invalid ledger schema at line {line_number}"
                        )
                    _validate_finite_tree(body, f"ledger[{line_number}]")
                    if body["prev_sha"] != expected_prev:
                        raise VirtualBrokerError(
                            f"ledger prev_sha mismatch at line {line_number}"
                        )
                    if not isinstance(supplied_sha, str) or supplied_sha != _sha(body):
                        raise VirtualBrokerError(
                            f"ledger sha mismatch at line {line_number}"
                        )
                    expected_prev = supplied_sha
                    event = record.get("event")
                    payload = record.get("payload")
                    if event == "QUOTE_BATCH_BEGIN" or (
                        event == "ACCOUNT_MARK"
                        and isinstance(payload, dict)
                        and payload.get("contract") == ACCOUNT_MARK_CONTRACT
                    ):
                        # The snapshot schema deliberately remains V2 for this
                        # fresh-replay milestone and therefore carries no MTM
                        # chain cursor.  Refuse to mint duplicate indices when
                        # a prior continuous-evidence ledger is reopened.
                        self._continuous_evidence_resume_forbidden = True
                    if isinstance(event, str) and (
                        event in _STATEFUL_LEDGER_EVENTS or event.startswith("EXIT")
                    ):
                        requires_state_restore = True
            self._prev_sha = expected_prev
        self._state_restore_required = requires_state_restore
        self._state_restore_verified = not requires_state_restore
        self._handle = self.ledger_path.open("a", encoding="utf-8")

    # ---- ledger ----------------------------------------------------------
    def _log(self, event: str, payload: dict[str, Any]) -> None:
        _validate_finite_tree(payload)
        body = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": payload,
            "prev_sha": self._prev_sha,
        }
        record = {**body, "sha": _sha(body)}
        self._handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self._handle.flush()
        if not self.fast_ledger:
            os.fsync(self._handle.fileno())
        self._prev_sha = record["sha"]

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq:06d}"

    def _require_state_mutation_allowed(self) -> None:
        """Keep the terminal account mark immutable until SESSION_STOP is sealed."""

        if self._mtm_terminal_emitted:
            raise VirtualBrokerError("broker state mutation cannot follow TERMINAL")

    # ---- continuous replay evidence ------------------------------------
    @staticmethod
    def _normalized_feed_pairs(feed_pairs: Any) -> list[str]:
        if not isinstance(feed_pairs, (list, tuple)) or not feed_pairs:
            raise VirtualBrokerError("feed_pairs must be a non-empty sequence")
        normalized: list[str] = []
        seen: set[str] = set()
        for pair in feed_pairs:
            if not isinstance(pair, str):
                raise VirtualBrokerError("feed_pairs must contain pair strings")
            _validate_pair(pair)
            if pair in seen:
                raise VirtualBrokerError(f"duplicate feed pair: {pair}")
            seen.add(pair)
            normalized.append(pair)
        return sorted(normalized)

    def _normalized_batch_quotes(self, quotes: Any) -> list[dict[str, Any]]:
        if not isinstance(quotes, (list, tuple)) or not quotes:
            raise VirtualBrokerError("quote batch must be a non-empty sequence")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in quotes:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise VirtualBrokerError(
                    "quote batch rows must be (pair, bid, ask, ts)"
                )
            pair, bid, ask, ts = item
            if not isinstance(pair, str):
                raise VirtualBrokerError("quote pair must be a string")
            if pair in seen:
                raise VirtualBrokerError(f"duplicate pair in quote batch: {pair}")
            seen.add(pair)
            clean_bid, clean_ask = self._validate_quote(pair, bid, ask, ts)
            normalized.append(
                {"pair": pair, "bid": clean_bid, "ask": clean_ask, "ts": ts}
            )
        return sorted(normalized, key=lambda row: row["pair"])

    @staticmethod
    def _validate_replay_coordinate(
        coordinate: dict[str, Any], quotes: list[dict[str, Any]]
    ) -> None:
        if set(coordinate) != _REPLAY_COORDINATE_KEYS:
            raise VirtualBrokerError("replay coordinate schema mismatch")
        if coordinate.get("mode") != "replay":
            raise VirtualBrokerError("account-mark coordinate mode must be replay")
        epoch = coordinate.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise VirtualBrokerError("replay coordinate epoch must be an integer")
        phase = coordinate.get("phase")
        if phase not in {"O", "H", "L", "C"}:
            raise VirtualBrokerError("replay coordinate phase is invalid")
        if coordinate.get("granularity") not in {"M1", "S5"}:
            raise VirtualBrokerError("replay coordinate granularity is invalid")
        if coordinate.get("intrabar") not in {"OHLC", "OLHC"}:
            raise VirtualBrokerError("replay coordinate intrabar is invalid")
        for quote in quotes:
            ts = str(quote["ts"])
            if VirtualBroker._quote_phase(ts) != phase:
                raise VirtualBrokerError(
                    "quote timestamp phase does not match replay coordinate"
                )
            try:
                instant = datetime.fromisoformat(ts.rsplit("#", 1)[0])
            except ValueError as exc:
                raise VirtualBrokerError(
                    "quote timestamp is not a replay coordinate instant"
                ) from exc
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise VirtualBrokerError(
                    "quote timestamp must be timezone-aware for replay evidence"
                )
            if instant.timestamp() != float(epoch):
                raise VirtualBrokerError(
                    "quote timestamp epoch does not match replay coordinate"
                )

    @staticmethod
    def _validate_receipt_digest(receipt: dict[str, Any]) -> None:
        if set(receipt) != _QUOTE_BATCH_RECEIPT_KEYS:
            raise VirtualBrokerError("quote batch receipt schema mismatch")
        if receipt.get("contract") != QUOTE_BATCH_CONTRACT:
            raise VirtualBrokerError("quote batch receipt contract mismatch")
        supplied = receipt.get("batch_sha256")
        body = {key: value for key, value in receipt.items() if key != "batch_sha256"}
        if not isinstance(supplied, str) or supplied != _sha(body):
            raise VirtualBrokerError("quote batch receipt digest mismatch")
        quotes = receipt.get("quotes")
        if receipt.get("quotes_sha256") != _sha(quotes):
            raise VirtualBrokerError("quote batch quotes digest mismatch")

    @staticmethod
    def _validate_mark_digest(mark: dict[str, Any]) -> None:
        if set(mark) != _ACCOUNT_MARK_KEYS:
            raise VirtualBrokerError("account mark schema mismatch")
        if mark.get("contract") != ACCOUNT_MARK_CONTRACT:
            raise VirtualBrokerError("account mark contract mismatch")
        supplied = mark.get("mark_sha256")
        body = {key: value for key, value in mark.items() if key != "mark_sha256"}
        if not isinstance(supplied, str) or supplied != _sha(body):
            raise VirtualBrokerError("account mark digest mismatch")
        for field_name in ("account", "positions", "orders", "quotes"):
            if mark.get(f"{field_name}_sha256") != _sha(mark.get(field_name)):
                raise VirtualBrokerError(f"account mark {field_name} digest mismatch")

    def _require_latest_batch_receipt(self, batch_receipt: Any) -> dict[str, Any]:
        if not isinstance(batch_receipt, dict):
            raise VirtualBrokerError("batch_receipt must be an object")
        supplied = _canonical_json_copy("batch_receipt", batch_receipt)
        self._validate_receipt_digest(supplied)
        if self._last_batch_receipt is None or supplied != self._last_batch_receipt:
            raise VirtualBrokerError(
                "batch_receipt is not the broker's latest immutable receipt"
            )
        return supplied

    def record_quote_batch_begin(
        self,
        quotes: list[tuple[str, float, float, str]],
        *,
        coordinate: dict[str, Any],
        feed_pairs: list[str],
    ) -> dict[str, Any]:
        """Commit one expected feed coordinate before its quotes are applied.

        The broker, rather than the runner, owns the sequence and previous
        digest.  The receipt is therefore safe to bind to the post-action
        account mark without accepting a caller-authored chain claim.
        """

        if self._continuous_evidence_resume_forbidden:
            raise VirtualBrokerError(
                "continuous evidence requires a fresh broker session"
            )
        if self._mtm_terminal_emitted:
            raise VirtualBrokerError("quote batch cannot follow a terminal mark")
        if not isinstance(coordinate, dict) or not coordinate:
            raise VirtualBrokerError("quote batch coordinate must be an object")
        canonical_coordinate = _canonical_json_copy("coordinate", coordinate)
        canonical_feed_pairs = self._normalized_feed_pairs(feed_pairs)
        canonical_quotes = self._normalized_batch_quotes(quotes)
        self._validate_replay_coordinate(canonical_coordinate, canonical_quotes)
        batch_pairs = [str(row["pair"]) for row in canonical_quotes]
        if (
            self._account_mark_chain_started
            and self._last_batch_receipt is not None
            and self._last_phase_batch_index
            != int(self._last_batch_receipt["batch_index"])
        ):
            raise VirtualBrokerError(
                "previous quote batch has no post-action PHASE account mark"
            )
        body = {
            "contract": QUOTE_BATCH_CONTRACT,
            "batch_index": self._batch_index,
            "coordinate": canonical_coordinate,
            "feed_pairs": canonical_feed_pairs,
            "batch_pairs": batch_pairs,
            "coverage_complete": batch_pairs == canonical_feed_pairs,
            "quotes": canonical_quotes,
            "quotes_sha256": _sha(canonical_quotes),
            "previous_batch_sha256": self._previous_batch_sha256,
        }
        receipt = {**body, "batch_sha256": _sha(body)}
        self._validate_receipt_digest(receipt)
        self._log("QUOTE_BATCH_BEGIN", receipt)
        self._last_batch_receipt = _canonical_json_copy("quote batch receipt", receipt)
        self._previous_batch_sha256 = receipt["batch_sha256"]
        self._last_batch_quote_seq_before = self._quote_seq
        self._last_batch_applied = False
        self._batch_index += 1
        return _canonical_json_copy("quote batch receipt", receipt)

    def _require_latest_batch_applied(self, receipt: dict[str, Any]) -> None:
        baseline = self._last_batch_quote_seq_before
        if baseline is None:
            raise VirtualBrokerError("quote batch application baseline is missing")
        if not self._last_batch_applied:
            raise VirtualBrokerError("committed quote batch has not been applied")
        expected_quotes = receipt["quotes"]
        expected_watermark = baseline + len(expected_quotes)
        if self._quote_seq != expected_watermark:
            raise VirtualBrokerError(
                "quote state does not contain exactly the committed batch"
            )
        for offset, expected in enumerate(expected_quotes, start=1):
            pair = str(expected["pair"])
            current = self.last_quotes.get(pair)
            if current != (expected["bid"], expected["ask"], expected["ts"]):
                raise VirtualBrokerError(
                    f"applied quote does not match batch receipt for {pair}"
                )
            if self._last_quote_sequences.get(pair) != baseline + offset:
                raise VirtualBrokerError(
                    f"applied quote sequence does not match batch receipt for {pair}"
                )
            if self._last_quote_watermarks.get(pair) != expected_watermark:
                raise VirtualBrokerError(
                    f"applied quote watermark does not match batch receipt for {pair}"
                )

    @staticmethod
    def _require_cursor_matches_coordinate(
        cursor: dict[str, Any], coordinate: dict[str, Any]
    ) -> None:
        """Bind runner progress to the broker-owned batch coordinate."""

        for key in ("mode", "epoch", "phase"):
            if key not in cursor or key not in coordinate:
                raise VirtualBrokerError(
                    f"feed_cursor and coordinate require matching {key}"
                )
            if cursor[key] != coordinate[key]:
                raise VirtualBrokerError(
                    f"feed_cursor {key} does not match batch coordinate"
                )

    def _mark_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "trade_id": pos.trade_id,
                "pair": pos.pair,
                "side": pos.side,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "opened_ts": pos.opened_ts,
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
            }
            for _, pos in sorted(self.positions.items())
        ]

    def _mark_orders(self) -> list[dict[str, Any]]:
        return [
            {
                "order_id": order.order_id,
                "pair": order.pair,
                "side": order.side,
                "units": order.units,
                "limit_price": order.limit_price,
                "tp_pips": order.tp_pips,
                "sl_pips": order.sl_pips,
                "kind": order.kind,
            }
            for _, order in sorted(self.orders.items())
        ]

    def _mark_quotes(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pair, quote in sorted(self.last_quotes.items()):
            sequence = self._last_quote_sequences.get(pair)
            watermark = self._last_quote_watermarks.get(pair)
            if sequence is None or watermark is None:
                raise VirtualBrokerError(
                    f"quote evidence is incomplete for account mark pair {pair}"
                )
            rows.append(
                {
                    "pair": pair,
                    "bid": quote[0],
                    "ask": quote[1],
                    "ts": quote[2],
                    "sequence": sequence,
                    "watermark": watermark,
                }
            )
        return rows

    def account_mark(
        self,
        kind: str,
        *,
        coordinate: Optional[dict[str, Any]] = None,
        batch_receipt: Optional[dict[str, Any]] = None,
        feed_cursor: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Record a broker-reconstructed START, PHASE, or TERMINAL MTM mark."""

        if self._continuous_evidence_resume_forbidden:
            raise VirtualBrokerError(
                "continuous evidence requires a fresh broker session"
            )
        if kind not in {"START", "PHASE", "TERMINAL"}:
            raise VirtualBrokerError("account mark kind is invalid")
        if self._mtm_terminal_emitted:
            raise VirtualBrokerError("account mark cannot follow TERMINAL")

        canonical_coordinate: dict[str, Any] | None = None
        canonical_cursor: dict[str, Any] | None = None
        bound_batch_index: int | None = None
        bound_batch_sha: str | None = None

        if kind == "START":
            if (
                self._mark_index != 0
                or self._batch_index != 0
                or coordinate is not None
                or batch_receipt is not None
                or feed_cursor is not None
                or self._quote_seq != 0
                or self.positions
                or self.orders
                or self.feed_cursor is not None
            ):
                raise VirtualBrokerError(
                    "START mark must precede all batches and carry no binding"
                )
        elif kind == "PHASE":
            if not self._account_mark_chain_started:
                raise VirtualBrokerError("PHASE mark requires a preceding START")
            receipt = self._require_latest_batch_receipt(batch_receipt)
            self._require_latest_batch_applied(receipt)
            bound_batch_index = int(receipt["batch_index"])
            bound_batch_sha = str(receipt["batch_sha256"])
            if bound_batch_index != self._last_phase_batch_index + 1:
                raise VirtualBrokerError(
                    "batch receipt is reused or skips a PHASE account mark"
                )
            receipt_coordinate = receipt["coordinate"]
            if coordinate is None:
                canonical_coordinate = _canonical_json_copy(
                    "coordinate", receipt_coordinate
                )
            else:
                canonical_coordinate = _canonical_json_copy("coordinate", coordinate)
                if canonical_coordinate != receipt_coordinate:
                    raise VirtualBrokerError(
                        "account mark coordinate does not match batch receipt"
                    )
            cursor_value = self.feed_cursor if feed_cursor is None else feed_cursor
            if not isinstance(cursor_value, dict):
                raise VirtualBrokerError("PHASE account mark requires feed_cursor")
            canonical_cursor = _canonical_json_copy("feed_cursor", cursor_value)
            self._require_cursor_matches_coordinate(
                canonical_cursor, canonical_coordinate
            )
        else:
            if not self._account_mark_chain_started:
                raise VirtualBrokerError("TERMINAL mark requires a preceding START")
            if coordinate is not None:
                raise VirtualBrokerError("TERMINAL mark coordinate must be null")
            if self._last_batch_receipt is not None:
                if batch_receipt is None:
                    raise VirtualBrokerError(
                        "TERMINAL requires the latest PHASE-marked batch receipt"
                    )
                receipt = self._require_latest_batch_receipt(batch_receipt)
                bound_batch_index = int(receipt["batch_index"])
                bound_batch_sha = str(receipt["batch_sha256"])
                if bound_batch_index != self._last_phase_batch_index:
                    raise VirtualBrokerError(
                        "TERMINAL may bind only the latest PHASE-marked receipt"
                    )
            elif batch_receipt is not None:
                raise VirtualBrokerError("TERMINAL has no quote batch to bind")
            cursor_value = self.feed_cursor if feed_cursor is None else feed_cursor
            if self._last_batch_receipt is not None:
                if not isinstance(cursor_value, dict):
                    raise VirtualBrokerError(
                        "TERMINAL requires a completed feed_cursor"
                    )
                canonical_cursor = _canonical_json_copy("feed_cursor", cursor_value)
                self._require_cursor_matches_coordinate(
                    canonical_cursor, receipt["coordinate"]
                )
                if canonical_cursor.get("completed") is not True:
                    raise VirtualBrokerError(
                        "TERMINAL feed_cursor must declare completed true"
                    )
            elif cursor_value is not None:
                if not isinstance(cursor_value, dict):
                    raise VirtualBrokerError("feed_cursor must be an object")
                canonical_cursor = _canonical_json_copy("feed_cursor", cursor_value)

        account = self.account()
        positions = self._mark_positions()
        orders = self._mark_orders()
        quotes = self._mark_quotes()
        body = {
            "contract": ACCOUNT_MARK_CONTRACT,
            "mark_index": self._mark_index,
            "kind": kind,
            "coordinate": canonical_coordinate,
            "batch_index": bound_batch_index,
            "batch_sha256": bound_batch_sha,
            "feed_cursor": canonical_cursor,
            "account": account,
            "account_sha256": _sha(account),
            "positions": positions,
            "positions_sha256": _sha(positions),
            "orders": orders,
            "orders_sha256": _sha(orders),
            "quotes": quotes,
            "quotes_sha256": _sha(quotes),
            "previous_mark_sha256": self._previous_mark_sha256,
        }
        mark = {**body, "mark_sha256": _sha(body)}
        self._validate_mark_digest(mark)
        self._log("ACCOUNT_MARK", mark)
        self._previous_mark_sha256 = mark["mark_sha256"]
        self._mark_index += 1
        if kind == "START":
            self._account_mark_chain_started = True
        elif kind == "PHASE":
            assert bound_batch_index is not None
            self._last_phase_batch_index = bound_batch_index
        else:
            self._mtm_terminal_emitted = True
        return _canonical_json_copy("account mark", mark)

    # ---- conversion / accounting ----------------------------------------
    @staticmethod
    def _quote_phase(ts: str) -> Optional[str]:
        if "#" not in ts:
            return None
        phase = ts.rsplit("#", 1)[1]
        return phase or None

    def _quote_as_of(
        self, pair: str, as_of_sequence: Optional[int] = None
    ) -> tuple[float, float, str, int]:
        """Return the newest observed quote no later than ``as_of_sequence``.

        Fill conversion is point-in-time evidence.  A later quote from another
        pair, even in the same replay epoch, must never reprice an earlier fill.
        The bounded history covers asynchronous live polling and all four
        replay phases without allowing a future quote to leak backwards.
        """

        history = self._quote_history.get(pair, [])
        if as_of_sequence is None:
            if history:
                return history[-1]
        else:
            for quote in reversed(history):
                if quote[3] <= as_of_sequence:
                    return quote
        if pair not in self.last_quotes:
            raise VirtualBrokerError(f"quote required for conversion pair {pair}")
        last_sequence = self._last_quote_sequences.get(pair)
        if last_sequence is not None and (
            as_of_sequence is None or last_sequence <= as_of_sequence
        ):
            bid, ask, ts = self.last_quotes[pair]
            return bid, ask, ts, last_sequence
        raise VirtualBrokerError(f"no {pair} conversion quote at or before fill quote")

    def _conversion_evidence(
        self,
        pair: str,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> dict[str, Any]:
        """Bind the exact quote(s) used to convert quote-currency P/L to JPY."""

        _validate_pair(pair)
        parts = pair.split("_")
        quote_ccy = parts[1]
        sources: list[tuple[str, tuple[float, float, str, int]]] = []
        if quote_ccy == "JPY":
            rate = 1.0
        elif quote_ccy == "USD":
            usd_jpy = self._quote_as_of("USD_JPY", as_of_sequence)
            sources.append(("USD_JPY", usd_jpy))
            rate = (usd_jpy[0] + usd_jpy[1]) / 2.0
        else:
            direct_pair = f"{quote_ccy}_JPY"
            try:
                direct = self._quote_as_of(direct_pair, as_of_sequence)
            except VirtualBrokerError:
                direct = None
            if direct is not None:
                sources.append((direct_pair, direct))
                rate = (direct[0] + direct[1]) / 2.0
            else:
                usd_jpy = self._quote_as_of("USD_JPY", as_of_sequence)
                via_pair = f"USD_{quote_ccy}"
                via_usd = self._quote_as_of(via_pair, as_of_sequence)
                via_mid = (via_usd[0] + via_usd[1]) / 2.0
                if via_mid <= 0:
                    raise VirtualBrokerError(f"invalid conversion quote for {via_pair}")
                sources.extend((("USD_JPY", usd_jpy), (via_pair, via_usd)))
                rate = ((usd_jpy[0] + usd_jpy[1]) / 2.0) / via_mid
        reference_epoch = (
            self._ts_epoch(reference_ts) if reference_ts is not None else None
        )
        for source_pair, source_quote in sources:
            source_ts = source_quote[2]
            if source_ts == reference_ts:
                continue
            source_epoch = self._ts_epoch(source_ts)
            if reference_ts is not None and (
                reference_epoch is None or source_epoch is None
            ):
                raise VirtualBrokerError(
                    f"unparseable conversion freshness timestamp for {source_pair}"
                )
            if reference_epoch is not None and source_epoch is not None:
                age_s = reference_epoch - source_epoch
                if abs(age_s) > MAX_CONVERSION_QUOTE_AGE_S:
                    raise VirtualBrokerError(
                        f"stale conversion quote for {source_pair}: {age_s:.3f}s"
                    )
        rate = _finite_number("conversion rate", rate, positive=True)
        return {
            "quote_currency": quote_ccy,
            "rate_jpy_per_quote_unit": rate,
            "as_of_quote_sequence": as_of_sequence,
            "source_quote_sequences": [quote[3] for _, quote in sources],
            "source_quotes": [
                {
                    "pair": source_pair,
                    "bid": quote[0],
                    "ask": quote[1],
                    "ts": quote[2],
                    "phase": self._quote_phase(quote[2]),
                }
                for source_pair, quote in sources
            ],
        }

    def _jpy_per_quote_unit(
        self,
        pair: str,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> float:
        """JPY value derived only from quotes observed by the requested time."""

        return float(
            self._conversion_evidence(pair, as_of_sequence, reference_ts)[
                "rate_jpy_per_quote_unit"
            ]
        )

    @staticmethod
    def _ts_epoch(ts: str) -> Optional[float]:
        try:
            return datetime.fromisoformat(ts.split("#")[0]).timestamp()
        except Exception:
            return None

    def _financing_jpy(
        self, pos: VBPosition, exit_ts: str, conversion_rate: Optional[float] = None
    ) -> float:
        if self.financing_pips_per_day <= 0:
            return 0.0
        t0 = self._ts_epoch(pos.opened_ts)
        t1 = self._ts_epoch(exit_ts)
        if t0 is None or t1 is None:
            raise VirtualBrokerError("financing requires parseable position timestamps")
        if t1 < t0:
            raise VirtualBrokerError("financing timestamp precedes position open")
        if t1 == t0:
            return 0.0
        days = (t1 - t0) / 86400.0
        rate = (
            self._jpy_per_quote_unit(pos.pair)
            if conversion_rate is None
            else conversion_rate
        )
        return self.financing_pips_per_day * _pip(pos.pair) * pos.units * rate * days

    def _position_pl_jpy(
        self,
        pos: VBPosition,
        bid: float,
        ask: float,
        *,
        as_of_sequence: Optional[int],
        reference_ts: str,
    ) -> tuple[float, float]:
        mark = bid if pos.side == "LONG" else ask
        diff = (
            (mark - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - mark)
        )
        rate = self._jpy_per_quote_unit(pos.pair, as_of_sequence, reference_ts)
        return diff * pos.units * rate, rate

    def _exposure_jpy(
        self,
        pair: str,
        units: float,
        price: float,
        *,
        as_of_sequence: Optional[int] = None,
        reference_ts: Optional[str] = None,
    ) -> float:
        if pair.endswith("JPY"):
            return units * price
        # base-currency exposure valued via quote conversion
        return (
            units * price * self._jpy_per_quote_unit(pair, as_of_sequence, reference_ts)
        )

    def _adverse_exit_price(self, pair: str, side: str, price: float) -> float:
        slip = self.slippage_pips * _pip(pair)
        exit_price = _round_price(
            pair, price - slip if side == "LONG" else price + slip
        )
        if exit_price <= 0:
            raise VirtualBrokerError("slippage produced a non-positive exit price")
        return exit_price

    def account(self) -> dict[str, Any]:
        equity = self.balance_jpy
        margin = 0.0
        accrued_financing = 0.0
        by_pair: dict[str, dict[str, float]] = {}
        for pos in self.positions.values():
            q = self.last_quotes.get(pos.pair)
            if q is None:
                raise VirtualBrokerError(f"no quote for open position pair {pos.pair}")
            watermark = self._last_quote_watermarks.get(pos.pair)
            if watermark is None:
                raise VirtualBrokerError(
                    f"no accounting watermark for open position pair {pos.pair}"
                )
            unrealized, conversion_rate = self._position_pl_jpy(
                pos,
                q[0],
                q[1],
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            financing = self._financing_jpy(pos, q[2], conversion_rate)
            equity += unrealized - financing
            accrued_financing += financing
            side_units = by_pair.setdefault(pos.pair, {"LONG": 0.0, "SHORT": 0.0})
            side_units[pos.side] += pos.units
        for pair, sides in by_pair.items():
            q = self.last_quotes[pair]
            mid = (q[0] + q[1]) / 2.0
            margin += (
                self._exposure_jpy(
                    pair,
                    max(sides["LONG"], sides["SHORT"]),
                    mid,
                    as_of_sequence=self._last_quote_watermarks[pair],
                    reference_ts=q[2],
                )
                / self.leverage
            )
        usage = margin / equity if equity > 0 else 999.0
        return {
            "balance_jpy": round(self.balance_jpy, 2),
            "equity_jpy": round(equity, 2),
            "margin_used_jpy": round(margin, 2),
            "margin_usage": round(usage, 6),
            "accrued_financing_jpy": round(accrued_financing, 2),
            "open_positions": len(self.positions),
            "resting_orders": len(self.orders),
        }

    def _margin_headroom_ok(self, pair: str, side: str, units: float) -> bool:
        """OANDA-faithful: refuse orders whose margin would not fit."""

        acct = self.account()
        q = self.last_quotes.get(pair)
        if q is None:
            return False
        mid = (q[0] + q[1]) / 2.0
        long_u = sum(
            p.units
            for p in self.positions.values()
            if p.pair == pair and p.side == "LONG"
        )
        short_u = sum(
            p.units
            for p in self.positions.values()
            if p.pair == pair and p.side == "SHORT"
        )
        if side == "LONG":
            long_u += units
        else:
            short_u += units
        watermark = self._last_quote_watermarks.get(pair)
        if watermark is None:
            raise VirtualBrokerError(f"no accounting watermark for {pair}")
        new_pair_margin = (
            self._exposure_jpy(
                pair,
                max(long_u, short_u),
                mid,
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            / self.leverage
        )
        old_pair_margin = (
            self._exposure_jpy(
                pair,
                max(
                    long_u - (units if side == "LONG" else 0),
                    short_u - (units if side == "SHORT" else 0),
                ),
                mid,
                as_of_sequence=watermark,
                reference_ts=q[2],
            )
            / self.leverage
        )
        new_total = acct["margin_used_jpy"] - old_pair_margin + new_pair_margin
        return new_total <= acct["equity_jpy"]

    def _entry_admission_rejection(
        self, pair: str, side: str, order_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Run an installed DOJO admission policy at the actual fill boundary.

        The base virtual broker has no strategy policy.  A provenance owner
        registry may install one, and a malformed decision fails closed before
        either a market or resting order can become a position.
        """

        if self._entry_admission is None:
            return None
        rejection = self._entry_admission(pair, side, order_id)
        if rejection is None:
            return None
        expected_keys = {
            "scope",
            "reason",
            "active_pair_positions",
            "max_concurrent_per_pair",
            "active_global_positions",
            "global_max_concurrent",
        }
        if not isinstance(rejection, dict) or set(rejection) != expected_keys:
            raise VirtualBrokerError(
                "entry admission policy returned malformed evidence"
            )
        if rejection["scope"] not in {"PAIR", "GLOBAL"} or rejection["reason"] not in {
            "OWNER_PAIR_CONCURRENCY_CAP_REACHED",
            "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED",
        }:
            raise VirtualBrokerError(
                "entry admission policy returned an invalid reason"
            )
        for key in ("active_pair_positions", "active_global_positions"):
            value = rejection[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VirtualBrokerError(
                    "entry admission policy returned an invalid count"
                )
        for key in ("max_concurrent_per_pair", "global_max_concurrent"):
            value = rejection[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VirtualBrokerError(
                    "entry admission policy returned an invalid cap"
                )
        return rejection

    # ---- agent actions ---------------------------------------------------
    def market_order(
        self,
        pair: str,
        side: str,
        units: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        self._require_state_mutation_allowed()
        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        q = self.last_quotes.get(pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pair}; cannot fill")
        rejection = self._entry_admission_rejection(pair, side)
        if rejection is not None:
            self._log(
                "ORDER_REJECTED_CONCURRENCY_CAP",
                {
                    "pair": pair,
                    "side": side,
                    "units": units,
                    "admission": rejection,
                },
            )
            raise VirtualBrokerError("owner concurrency cap reached for market order")
        if not self._margin_headroom_ok(pair, side, units):
            self._log(
                "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                {"pair": pair, "side": side, "units": units},
            )
            raise VirtualBrokerError("insufficient margin for market order")
        bid, ask, ts = q
        quote_sequence = self._last_quote_watermarks.get(pair)
        if quote_sequence is None:
            raise VirtualBrokerError(f"no accounting watermark for {pair}")
        conversion = self._conversion_evidence(pair, quote_sequence, ts)
        pip = _pip(pair)
        slip = self.slippage_pips * pip
        entry = (ask + slip) if side == "LONG" else (bid - slip)
        entry = _round_price(pair, entry)
        if entry <= 0:
            raise VirtualBrokerError("slippage produced a non-positive entry price")
        tp = (
            _round_price(
                pair, entry + tp_pips * pip if side == "LONG" else entry - tp_pips * pip
            )
            if tp_pips
            else None
        )
        sl = (
            _round_price(
                pair, entry - sl_pips * pip if side == "LONG" else entry + sl_pips * pip
            )
            if sl_pips
            else None
        )
        trade_id = self._next_id("T")
        self.positions[trade_id] = VBPosition(
            trade_id=trade_id,
            pair=pair,
            side=side,
            units=units,
            entry_price=entry,
            opened_ts=ts,
            tp_price=tp,
            sl_price=sl,
        )
        self._log(
            "FILL_MARKET",
            {
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "units": units,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            },
        )
        # Attached exits are atomic with a market fill.  If the currently
        # executable exit-side quote already crossed the SL/TP (for example a
        # spread wider than the stop), resolve it on this same quote instead
        # of carrying an impossible position until a future tick.
        self._resolve_attached_exit_at_quote(trade_id, bid, ask, ts, quote_sequence)
        self._enforce_margin_after_action()
        return trade_id

    def limit_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        self._require_state_mutation_allowed()
        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        price = _finite_number("price", price, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        order_id = self._next_id("O")
        self.orders[order_id] = VBOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            units=units,
            limit_price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
        self._log(
            "ORDER_LIMIT",
            {
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "units": units,
                "price": price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
            },
        )
        return order_id

    def stop_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: Optional[float] = None,
        sl_pips: Optional[float] = None,
    ) -> str:
        """Breakout entry: LONG fills once the real ask reaches price (at
        the level or WORSE when gapped); SHORT once the real bid does."""

        self._require_state_mutation_allowed()
        _validate_pair(pair)
        if side not in {"LONG", "SHORT"}:
            raise VirtualBrokerError(f"invalid side: {side}")
        units = _finite_number("units", units, positive=True)
        price = _finite_number("price", price, positive=True)
        if tp_pips is not None:
            tp_pips = _finite_number("tp_pips", tp_pips, positive=True)
        if sl_pips is not None:
            sl_pips = _finite_number("sl_pips", sl_pips, positive=True)
        order_id = self._next_id("O")
        self.orders[order_id] = VBOrder(
            order_id=order_id,
            pair=pair,
            side=side,
            units=units,
            limit_price=price,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            kind="STOP",
        )
        self._log(
            "ORDER_STOP",
            {
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "units": units,
                "price": price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
            },
        )
        return order_id

    def cancel_order(self, order_id: str) -> None:
        self._require_state_mutation_allowed()
        if order_id not in self.orders:
            raise VirtualBrokerError(f"unknown order: {order_id}")
        del self.orders[order_id]
        self._log("ORDER_CANCEL", {"order_id": order_id})

    def close_trade(self, trade_id: str, units: Optional[float] = None) -> float:
        self._require_state_mutation_allowed()
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        q = self.last_quotes.get(pos.pair)
        if q is None:
            raise VirtualBrokerError(f"no live quote for {pos.pair}; cannot close")
        bid, ask, ts = q
        quote_sequence = self._last_quote_watermarks.get(pos.pair)
        if quote_sequence is None:
            raise VirtualBrokerError(f"no accounting watermark for {pos.pair}")
        conversion = self._conversion_evidence(pos.pair, quote_sequence, ts)
        conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
        requested_units = (
            pos.units
            if units is None
            else _finite_number("close units", units, positive=True)
        )
        close_units = min(requested_units, pos.units)
        price = self._adverse_exit_price(
            pos.pair, pos.side, bid if pos.side == "LONG" else ask
        )
        diff = (
            (price - pos.entry_price)
            if pos.side == "LONG"
            else (pos.entry_price - price)
        )
        gross_pl = diff * close_units * conversion_rate
        financing = self._financing_jpy(pos, ts, conversion_rate) * (
            close_units / pos.units
        )
        pl = gross_pl - financing
        self.balance_jpy += pl
        if close_units >= pos.units:
            del self.positions[trade_id]
        else:
            pos.units -= close_units
        self._log(
            "CLOSE",
            {
                "trade_id": trade_id,
                "units": close_units,
                "price": price,
                "pl_jpy": round(pl, 2),
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "gross_pl_jpy": round(gross_pl, 2),
                "financing_jpy": round(financing, 2),
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            },
        )
        return pl

    def set_exit(
        self,
        trade_id: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> None:
        self._require_state_mutation_allowed()
        pos = self.positions.get(trade_id)
        if pos is None:
            raise VirtualBrokerError(f"unknown trade: {trade_id}")
        if tp_price is not None:
            tp_price = _finite_number("tp_price", tp_price, positive=True)
        if sl_price is not None:
            sl_price = _finite_number("sl_price", sl_price, positive=True)
        pos.tp_price = tp_price
        pos.sl_price = sl_price
        self._log("SET_EXIT", {"trade_id": trade_id, "tp": tp_price, "sl": sl_price})

    # ---- feed ------------------------------------------------------------
    def on_quote(
        self, pair: str, bid: float, ask: float, ts: str
    ) -> list[dict[str, Any]]:
        """Process one real quote: resting orders, TP/SL, margin. Returns events."""

        self._require_state_mutation_allowed()
        if self._account_mark_chain_started and not self._mtm_terminal_emitted:
            raise VirtualBrokerError(
                "continuous replay evidence requires atomic committed quote batches"
            )
        bid, ask = self._validate_quote(pair, bid, ask, ts)
        self._record_quote(pair, bid, ask, ts)
        self._last_quote_watermarks[pair] = self._quote_seq
        return self._process_current_quote(pair, bid, ask, ts, self._quote_seq)

    def on_quote_batch(
        self, quotes: list[tuple[str, float, float, str]]
    ) -> list[dict[str, Any]]:
        """Atomically stage a simultaneous feed phase before processing fills.

        Conversion rates and margin marks then use one batch watermark rather
        than depending on lexicographic pair delivery order.
        """

        self._require_state_mutation_allowed()
        if not quotes:
            raise VirtualBrokerError("quote batch must not be empty")
        if self._account_mark_chain_started:
            if self._last_batch_receipt is None:
                raise VirtualBrokerError(
                    "quote batch has no preceding broker commitment"
                )
            if (
                self._last_phase_batch_index
                == int(self._last_batch_receipt["batch_index"])
                or self._last_batch_applied
            ):
                raise VirtualBrokerError(
                    "quote batch has no unconsumed broker commitment"
                )
            supplied_quotes = self._normalized_batch_quotes(quotes)
            if supplied_quotes != self._last_batch_receipt["quotes"]:
                raise VirtualBrokerError(
                    "applied quote batch does not match broker commitment"
                )
            if self._quote_seq != self._last_batch_quote_seq_before:
                raise VirtualBrokerError(
                    "quote state advanced outside the committed batch"
                )
        normalized: list[tuple[str, float, float, str]] = []
        seen: set[str] = set()
        for pair, bid, ask, ts in quotes:
            if pair in seen:
                raise VirtualBrokerError(f"duplicate pair in quote batch: {pair}")
            seen.add(pair)
            clean_bid, clean_ask = self._validate_quote(pair, bid, ask, ts)
            normalized.append((pair, clean_bid, clean_ask, ts))
        # A simultaneous batch has no caller-defined priority.  Canonicalize
        # the tie-break so capital competition and ledger order cannot change
        # merely because the same quotes were supplied in a different list
        # order.
        normalized.sort(key=lambda item: item[0])
        for pair, bid, ask, ts in normalized:
            self._record_quote(pair, bid, ask, ts)
        batch_watermark = self._quote_seq
        for pair, _, _, _ in normalized:
            self._last_quote_watermarks[pair] = batch_watermark
        events: list[dict[str, Any]] = []
        for pair, bid, ask, ts in normalized:
            events.extend(
                self._process_current_quote(
                    pair,
                    bid,
                    ask,
                    ts,
                    batch_watermark,
                    enforce_margin=False,
                )
            )
        # Every quote in the atomic phase and every ordinary fill/exit must be
        # visible before deciding a portfolio-level margin closeout.  Running
        # this per pair makes a TP on one pair rescue (or fail to rescue) the
        # account solely according to iteration order.
        events.extend(self._enforce_margin_after_action())
        if self._account_mark_chain_started:
            self._last_batch_applied = True
        return events

    @staticmethod
    def _validate_quote(
        pair: str, bid: float, ask: float, ts: str
    ) -> tuple[float, float]:
        _validate_pair(pair)
        bid = _finite_number("bid", bid, positive=True)
        ask = _finite_number("ask", ask, positive=True)
        if ask < bid:
            raise VirtualBrokerError(f"invalid quote {pair} {bid}/{ask}")
        if not isinstance(ts, str) or not ts:
            raise VirtualBrokerError("quote timestamp must be a non-empty string")
        return bid, ask

    def _record_quote(self, pair: str, bid: float, ask: float, ts: str) -> None:
        self._quote_seq += 1
        self.last_quotes[pair] = (bid, ask, ts)
        self._last_quote_sequences[pair] = self._quote_seq
        history = self._quote_history.setdefault(pair, [])
        history.append((bid, ask, ts, self._quote_seq))
        # Four replay phases plus asynchronous cross-pair polling need only a
        # short tail.  Keep a wider fixed bound so long sessions cannot grow
        # without limit while conversion remains point-in-time reproducible.
        if len(history) > 128:
            del history[:-128]

    def _resolve_attached_exit_at_quote(
        self,
        trade_id: str,
        bid: float,
        ask: float,
        ts: str,
        quote_sequence: int,
    ) -> dict[str, Any] | None:
        """Resolve one attached SL/TP against the already-executable quote."""

        pos = self.positions.get(trade_id)
        if pos is None:
            return None
        exit_price = None
        reason = None
        if pos.side == "LONG":
            if pos.sl_price is not None and bid <= pos.sl_price:
                exit_price, reason = min(pos.sl_price, bid), "SL"
            elif pos.tp_price is not None and bid >= pos.tp_price:
                exit_price, reason = pos.tp_price, "TP"
        else:
            if pos.sl_price is not None and ask >= pos.sl_price:
                exit_price, reason = max(pos.sl_price, ask), "SL"
            elif pos.tp_price is not None and ask <= pos.tp_price:
                exit_price, reason = pos.tp_price, "TP"
        if exit_price is None or reason is None:
            return None
        applied_slippage_pips = 0.0
        if reason == "SL":
            stressed_exit = self._adverse_exit_price(pos.pair, pos.side, exit_price)
            applied_slippage_pips = abs(stressed_exit - exit_price) / _pip(pos.pair)
            exit_price = stressed_exit
        else:
            # TP is a price-protected limit exit.  Fixed stress slippage must
            # not fabricate an execution through its protected price.
            exit_price = _round_price(pos.pair, exit_price)
        conversion = self._conversion_evidence(pos.pair, quote_sequence, ts)
        conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
        diff = (
            (exit_price - pos.entry_price)
            if pos.side == "LONG"
            else (pos.entry_price - exit_price)
        )
        gross_pl = diff * pos.units * conversion_rate
        financing = self._financing_jpy(pos, ts, conversion_rate)
        pl = gross_pl - financing
        self.balance_jpy += pl
        del self.positions[trade_id]
        event = {
            "event": f"EXIT_{reason}",
            "trade_id": trade_id,
            "price": exit_price,
            "pl_jpy": round(pl, 2),
            "quote": {"bid": bid, "ask": ask, "ts": ts},
            "gross_pl_jpy": round(gross_pl, 2),
            "financing_jpy": round(financing, 2),
            "conversion": conversion,
            "slippage_pips": self.slippage_pips,
            "applied_slippage_pips": round(applied_slippage_pips, 8),
            "price_protection": reason == "TP",
        }
        self._log(f"EXIT_{reason}", event)
        return event

    def _process_current_quote(
        self,
        pair: str,
        bid: float,
        ask: float,
        ts: str,
        quote_sequence: int,
        *,
        enforce_margin: bool = True,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        # resting limit fills (worse-of level/quote when gapped)
        for order_id in list(self.orders):
            order = self.orders[order_id]
            if order.pair != pair:
                continue
            filled_price = None
            if order.kind == "LIMIT":
                if order.side == "LONG" and ask <= order.limit_price:
                    filled_price = min(order.limit_price, ask)
                elif order.side == "SHORT" and bid >= order.limit_price:
                    filled_price = max(order.limit_price, bid)
            else:  # STOP: triggers at the level, fills at level or worse
                if order.side == "LONG" and ask >= order.limit_price:
                    filled_price = max(order.limit_price, ask)
                elif order.side == "SHORT" and bid <= order.limit_price:
                    filled_price = min(order.limit_price, bid)
            if filled_price is None:
                continue
            rejection = self._entry_admission_rejection(pair, order.side, order_id)
            if rejection is not None:
                del self.orders[order_id]
                event = {
                    "event": "ORDER_CANCEL_CONCURRENCY_CAP",
                    "order_id": order_id,
                    "pair": pair,
                    "side": order.side,
                    "units": order.units,
                    "quote": {"bid": bid, "ask": ask, "ts": ts},
                    "admission": rejection,
                }
                self._log("ORDER_CANCEL_CONCURRENCY_CAP", event)
                events.append(event)
                continue
            applied_slippage_pips = 0.0
            if self.slippage_pips > 0:
                slip = self.slippage_pips * _pip(pair)
                stressed_price = _round_price(
                    pair,
                    filled_price + slip
                    if order.side == "LONG"
                    else filled_price - slip,
                )
                if order.kind == "LIMIT":
                    # A limit order may receive less improvement under stress,
                    # but can never execute through its protected price.
                    filled_price = (
                        min(order.limit_price, stressed_price)
                        if order.side == "LONG"
                        else max(order.limit_price, stressed_price)
                    )
                else:
                    filled_price = stressed_price
                if filled_price <= 0:
                    raise VirtualBrokerError(
                        "slippage produced a non-positive entry price"
                    )
                observed_price = ask if order.side == "LONG" else bid
                applied_slippage_pips = abs(filled_price - observed_price) / _pip(pair)
            if not self._margin_headroom_ok(pair, order.side, order.units):
                del self.orders[order_id]
                self._log(
                    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                    {"order_id": order_id, "pair": pair},
                )
                continue
            pip = _pip(pair)
            tp = (
                _round_price(
                    pair,
                    filled_price + order.tp_pips * pip
                    if order.side == "LONG"
                    else filled_price - order.tp_pips * pip,
                )
                if order.tp_pips
                else None
            )
            sl = (
                _round_price(
                    pair,
                    filled_price - order.sl_pips * pip
                    if order.side == "LONG"
                    else filled_price + order.sl_pips * pip,
                )
                if order.sl_pips
                else None
            )
            trade_id = self._next_id("T")
            conversion = self._conversion_evidence(pair, quote_sequence, ts)
            self.positions[trade_id] = VBPosition(
                trade_id=trade_id,
                pair=pair,
                side=order.side,
                units=order.units,
                entry_price=filled_price,
                opened_ts=ts,
                tp_price=tp,
                sl_price=sl,
            )
            del self.orders[order_id]
            event = {
                "event": "FILL_LIMIT",
                "order_id": order_id,
                "trade_id": trade_id,
                "pair": pair,
                "side": order.side,
                "units": order.units,
                "price": filled_price,
                "entry": filled_price,
                "tp": tp,
                "sl": sl,
                "order_kind": order.kind,
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
                "applied_slippage_pips": round(applied_slippage_pips, 8),
                "price_protection": order.kind == "LIMIT",
            }
            self._log("FILL_LIMIT", event)
            events.append(event)

        # TP/SL: SL first when both touch on the same quote (pessimistic)
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            if pos.pair != pair:
                continue
            event = self._resolve_attached_exit_at_quote(
                trade_id, bid, ask, ts, quote_sequence
            )
            if event is not None:
                events.append(event)

        if enforce_margin:
            events.extend(self._enforce_margin_after_action())
        return events

    def _enforce_margin_after_action(self) -> list[dict[str, Any]]:
        acct = self.account()
        if acct["margin_usage"] < CLOSEOUT_USAGE or not self.positions:
            return []
        events = []
        for trade_id in list(self.positions):
            pos = self.positions[trade_id]
            q = self.last_quotes[pos.pair]
            quote_sequence = self._last_quote_watermarks.get(pos.pair)
            if quote_sequence is None:
                raise VirtualBrokerError(
                    f"no accounting watermark for open position pair {pos.pair}"
                )
            conversion = self._conversion_evidence(pos.pair, quote_sequence, q[2])
            conversion_rate = float(conversion["rate_jpy_per_quote_unit"])
            price = self._adverse_exit_price(
                pos.pair, pos.side, q[0] if pos.side == "LONG" else q[1]
            )
            diff = (
                (price - pos.entry_price)
                if pos.side == "LONG"
                else (pos.entry_price - price)
            )
            gross_pl = diff * pos.units * conversion_rate
            financing = self._financing_jpy(pos, q[2], conversion_rate)
            pl = gross_pl - financing
            self.balance_jpy += pl
            del self.positions[trade_id]
            event = {
                "event": "MARGIN_CLOSEOUT",
                "trade_id": trade_id,
                "price": price,
                "pl_jpy": round(pl, 2),
                "gross_pl_jpy": round(gross_pl, 2),
                "financing_jpy": round(financing, 2),
                "quote": {"bid": q[0], "ask": q[1], "ts": q[2]},
                "conversion": conversion,
                "slippage_pips": self.slippage_pips,
            }
            self._log("MARGIN_CLOSEOUT", event)
            events.append(event)
        return events

    # ---- persistence -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        snap = {
            "schema": SNAPSHOT_SCHEMA,
            "balance_jpy": self.balance_jpy,
            "seq": self._seq,
            "positions": [vars(p) for p in self.positions.values()],
            "orders": [vars(o) for o in self.orders.values()],
            "quote_seq": self._quote_seq,
            "last_quotes": {
                pair: {"bid": q[0], "ask": q[1], "ts": q[2]}
                for pair, q in self.last_quotes.items()
            },
            "last_quote_sequences": dict(self._last_quote_sequences),
            "last_quote_watermarks": dict(self._last_quote_watermarks),
            "quote_history": {
                pair: [
                    {"bid": q[0], "ask": q[1], "ts": q[2], "sequence": q[3]}
                    for q in history
                ]
                for pair, history in self._quote_history.items()
            },
            "feed_cursor": self.feed_cursor,
            "ledger_tip_sha": self._prev_sha,
        }
        _validate_finite_tree(snap, "snapshot")
        return snap

    def restore(self, snap: dict[str, Any]) -> None:
        self._require_state_mutation_allowed()
        if not isinstance(snap, dict):
            raise VirtualBrokerError("broker snapshot must be an object")
        if set(snap) != _SNAPSHOT_KEYS or snap.get("schema") != SNAPSHOT_SCHEMA:
            raise VirtualBrokerError("broker snapshot schema mismatch")
        _validate_finite_tree(snap, "snapshot")
        balance_jpy = _finite_number("snapshot balance_jpy", snap["balance_jpy"])
        seq_value = _finite_number("snapshot seq", snap["seq"], non_negative=True)
        if not seq_value.is_integer():
            raise VirtualBrokerError("snapshot seq must be an integer")
        sequence_counter = int(seq_value)

        positions: dict[str, VBPosition] = {}
        raw_positions = snap.get("positions")
        if not isinstance(raw_positions, list):
            raise VirtualBrokerError("snapshot positions must be a list")
        for raw in raw_positions:
            if not isinstance(raw, dict):
                raise VirtualBrokerError("snapshot position must be an object")
            try:
                pos = VBPosition(**raw)
            except (KeyError, TypeError) as exc:
                raise VirtualBrokerError("invalid snapshot position schema") from exc
            _validate_pair(pos.pair)
            if pos.side not in {"LONG", "SHORT"}:
                raise VirtualBrokerError("invalid snapshot position side")
            pos.units = _finite_number(
                "snapshot position units", pos.units, positive=True
            )
            pos.entry_price = _finite_number(
                "snapshot position entry_price", pos.entry_price, positive=True
            )
            if pos.tp_price is not None:
                pos.tp_price = _finite_number(
                    "snapshot position tp_price", pos.tp_price, positive=True
                )
            if pos.sl_price is not None:
                pos.sl_price = _finite_number(
                    "snapshot position sl_price", pos.sl_price, positive=True
                )
            if not pos.trade_id or pos.trade_id in positions:
                raise VirtualBrokerError("duplicate or empty snapshot trade_id")
            if not isinstance(pos.opened_ts, str) or not pos.opened_ts:
                raise VirtualBrokerError("snapshot position opened_ts is required")
            positions[pos.trade_id] = pos

        orders: dict[str, VBOrder] = {}
        raw_orders = snap.get("orders")
        if not isinstance(raw_orders, list):
            raise VirtualBrokerError("snapshot orders must be a list")
        for raw in raw_orders:
            if not isinstance(raw, dict):
                raise VirtualBrokerError("snapshot order must be an object")
            try:
                order = VBOrder(**raw)
            except (KeyError, TypeError) as exc:
                raise VirtualBrokerError("invalid snapshot order schema") from exc
            _validate_pair(order.pair)
            if order.side not in {"LONG", "SHORT"} or order.kind not in {
                "LIMIT",
                "STOP",
            }:
                raise VirtualBrokerError("invalid snapshot order side/kind")
            order.units = _finite_number(
                "snapshot order units", order.units, positive=True
            )
            order.limit_price = _finite_number(
                "snapshot order price", order.limit_price, positive=True
            )
            if order.tp_pips is not None:
                order.tp_pips = _finite_number(
                    "snapshot order tp_pips", order.tp_pips, positive=True
                )
            if order.sl_pips is not None:
                order.sl_pips = _finite_number(
                    "snapshot order sl_pips", order.sl_pips, positive=True
                )
            if not order.order_id or order.order_id in orders:
                raise VirtualBrokerError("duplicate or empty snapshot order_id")
            orders[order.order_id] = order

        raw_last_quotes = snap.get("last_quotes", {})
        raw_sequences = snap.get("last_quote_sequences", {})
        raw_watermarks = snap.get("last_quote_watermarks", {})
        raw_history = snap.get("quote_history", {})
        if not all(
            isinstance(value, dict)
            for value in (raw_last_quotes, raw_sequences, raw_watermarks, raw_history)
        ):
            raise VirtualBrokerError("invalid snapshot quote state")
        quote_seq_value = _finite_number(
            "snapshot quote_seq", snap.get("quote_seq", 0), non_negative=True
        )
        if not quote_seq_value.is_integer():
            raise VirtualBrokerError("snapshot quote_seq must be an integer")
        quote_seq = int(quote_seq_value)
        last_quotes: dict[str, tuple[float, float, str]] = {}
        last_sequences: dict[str, int] = {}
        last_watermarks: dict[str, int] = {}
        quote_history: dict[str, list[tuple[float, float, str, int]]] = {}
        for pair, raw_quote in raw_last_quotes.items():
            if not isinstance(raw_quote, dict):
                raise VirtualBrokerError("invalid snapshot last quote")
            bid, ask = self._validate_quote(
                pair, raw_quote.get("bid"), raw_quote.get("ask"), raw_quote.get("ts")
            )
            sequence_value = _finite_number(
                "snapshot quote sequence", raw_sequences.get(pair), positive=True
            )
            watermark_value = _finite_number(
                "snapshot quote watermark", raw_watermarks.get(pair), positive=True
            )
            if not sequence_value.is_integer() or not watermark_value.is_integer():
                raise VirtualBrokerError("snapshot quote sequence must be an integer")
            sequence = int(sequence_value)
            watermark = int(watermark_value)
            if sequence > watermark or watermark > quote_seq:
                raise VirtualBrokerError("snapshot quote watermark is inconsistent")
            last_quotes[pair] = (bid, ask, raw_quote["ts"])
            last_sequences[pair] = sequence
            last_watermarks[pair] = watermark
            raw_pair_history = raw_history.get(pair)
            if not isinstance(raw_pair_history, list) or not raw_pair_history:
                raise VirtualBrokerError("snapshot quote history is missing")
            parsed_history: list[tuple[float, float, str, int]] = []
            previous_sequence = 0
            for raw_item in raw_pair_history:
                if not isinstance(raw_item, dict):
                    raise VirtualBrokerError("invalid snapshot quote history")
                hist_bid, hist_ask = self._validate_quote(
                    pair,
                    raw_item.get("bid"),
                    raw_item.get("ask"),
                    raw_item.get("ts"),
                )
                hist_sequence_value = _finite_number(
                    "snapshot history sequence",
                    raw_item.get("sequence"),
                    positive=True,
                )
                if not hist_sequence_value.is_integer():
                    raise VirtualBrokerError(
                        "snapshot history sequence must be an integer"
                    )
                hist_sequence = int(hist_sequence_value)
                if hist_sequence <= previous_sequence or hist_sequence > quote_seq:
                    raise VirtualBrokerError("snapshot quote history is not monotonic")
                previous_sequence = hist_sequence
                parsed_history.append(
                    (hist_bid, hist_ask, raw_item["ts"], hist_sequence)
                )
            if (
                parsed_history[-1][:3] != last_quotes[pair]
                or parsed_history[-1][3] != sequence
            ):
                raise VirtualBrokerError("snapshot last quote/history mismatch")
            quote_history[pair] = parsed_history
        if set(raw_sequences) != set(last_quotes) or set(raw_watermarks) != set(
            last_quotes
        ):
            raise VirtualBrokerError("snapshot quote maps disagree")
        if set(raw_history) != set(last_quotes):
            raise VirtualBrokerError("snapshot quote history pairs disagree")

        ledger_tip = snap.get("ledger_tip_sha")
        if (
            not isinstance(ledger_tip, str)
            or len(ledger_tip) != 64
            or any(char not in "0123456789abcdef" for char in ledger_tip)
            or ledger_tip != self._prev_sha
        ):
            raise VirtualBrokerError("snapshot ledger tip does not match ledger")
        feed_cursor = snap.get("feed_cursor")
        if feed_cursor is not None and not isinstance(feed_cursor, dict):
            raise VirtualBrokerError("snapshot feed_cursor must be an object")

        generated_ids = [*positions, *orders]
        generated_sequences: list[int] = []
        for identity in generated_ids:
            if (
                len(identity) != 7
                or identity[0] not in {"T", "O"}
                or not identity[1:].isdigit()
            ):
                raise VirtualBrokerError("snapshot contains an invalid generated id")
            generated_sequences.append(int(identity[1:]))
        if generated_sequences and sequence_counter < max(generated_sequences):
            raise VirtualBrokerError("snapshot sequence would reuse an existing id")

        self.balance_jpy = balance_jpy
        self._seq = sequence_counter
        self.positions = positions
        self.orders = orders
        self.last_quotes = last_quotes
        self._last_quote_sequences = last_sequences
        self._last_quote_watermarks = last_watermarks
        self._quote_history = quote_history
        self._quote_seq = quote_seq
        self.feed_cursor = feed_cursor
        self._state_restore_verified = True
