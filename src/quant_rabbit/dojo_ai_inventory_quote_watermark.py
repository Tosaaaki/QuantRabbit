"""Append-only executable quote watermarks for paper-AI inventory rooms.

The writer owns both immutable quote-source bytes and the room-local hash
chain.  Callers provide the observed quote values plus the digest of the
read-only acquisition receipt.  This module has no broker/order import and
cannot mutate inventory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_replay_lifecycle import canonical_paper_ai_rooms_root


QUOTE_WATERMARK_CONTRACT = "QR_DOJO_AI_INVENTORY_QUOTE_WATERMARK_V1"
QUOTE_SOURCE_CONTRACT = "QR_DOJO_AI_INVENTORY_QUOTE_SOURCE_V1"
QUOTE_WATERMARK_LEDGER_NAME = "quote_watermarks.jsonl"
QUOTE_SOURCE_DIRECTORY = "quote_sources"
GENESIS_QUOTE_SHA256 = "0" * 64
MAX_LEDGER_BYTES = 256 * 1024 * 1024
MAX_LEDGER_ROWS = 1_000_000
MAX_LINE_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 64 * 1024
MAX_QUOTE_AGE_SECONDS = 180

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_ROW_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "recorded_at_utc",
        "timestamp_utc",
        "pair",
        "bid",
        "ask",
        "source_sha256",
        "capture_source_sha256",
        "acquisition_receipt_sha256",
        "slippage_pips_per_fill",
        "financing_pips_per_day",
        "previous_quote_sha256",
        "quote_sha256",
        "paper_only",
        "order_authority",
        "live_permission",
    }
)


class AiInventoryQuoteWatermarkError(RuntimeError):
    """The quote watermark could not be trusted or appended."""


class AiInventoryQuoteWatermarkConflictError(AiInventoryQuoteWatermarkError):
    """The same pair/timestamp already has different immutable quote truth."""


class AiInventoryQuoteWatermarkMarketClosedError(AiInventoryQuoteWatermarkError):
    """A quote append was attempted while the deterministic FX week was closed."""


def append_ai_inventory_quote_watermark(
    room_root: Path,
    *,
    pair: str,
    bid: float,
    ask: float,
    timestamp_utc: str,
    slippage_pips_per_fill: float,
    financing_pips_per_day: float,
    acquisition_receipt_sha256: str,
    capture_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist one immutable quote source and append its room-local watermark.

    The market-open check intentionally precedes all filesystem reads.  A
    weekend invocation therefore performs neither source reconstruction nor a
    quote-ledger evaluation.
    """

    now = _utc_now().astimezone(timezone.utc)
    _require_market_open(now)
    root = _require_canonical_room_root(room_root)
    quote_at = _parse_utc(timestamp_utc, "timestamp_utc")
    _require_market_open(quote_at)
    if quote_at > now or (now - quote_at).total_seconds() > MAX_QUOTE_AGE_SECONDS:
        raise AiInventoryQuoteWatermarkError("quote is stale or future-dated")
    if not isinstance(pair, str) or _PAIR_RE.fullmatch(pair) is None:
        raise AiInventoryQuoteWatermarkError("pair is invalid")
    normalized_bid = _finite_nonnegative(bid, "bid", positive=True)
    normalized_ask = _finite_nonnegative(ask, "ask", positive=True)
    if normalized_ask < normalized_bid:
        raise AiInventoryQuoteWatermarkError("ask is below bid")
    slippage = _finite_nonnegative(
        slippage_pips_per_fill, "slippage_pips_per_fill"
    )
    financing = _finite_nonnegative(financing_pips_per_day, "financing_pips_per_day")
    if (
        not isinstance(acquisition_receipt_sha256, str)
        or _SHA256_RE.fullmatch(acquisition_receipt_sha256) is None
        or acquisition_receipt_sha256 == GENESIS_QUOTE_SHA256
    ):
        raise AiInventoryQuoteWatermarkError(
            "acquisition_receipt_sha256 is invalid"
        )
    if capture_source_sha256 is None:
        # Compatibility for isolated unit fixtures. Production captured-quote
        # ingestion always supplies the independently verified canonical
        # source digest.
        capture_source_sha256 = acquisition_receipt_sha256
    if (
        not isinstance(capture_source_sha256, str)
        or _SHA256_RE.fullmatch(capture_source_sha256) is None
        or capture_source_sha256 == GENESIS_QUOTE_SHA256
    ):
        raise AiInventoryQuoteWatermarkError("capture_source_sha256 is invalid")

    source = {
        "contract": QUOTE_SOURCE_CONTRACT,
        "timestamp_utc": _format_utc(quote_at),
        "pair": pair,
        "bid": normalized_bid,
        "ask": normalized_ask,
        "capture_source_sha256": capture_source_sha256,
        "acquisition_receipt_sha256": acquisition_receipt_sha256,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    source_raw = _canonical_json_bytes(source) + b"\n"
    source_sha = hashlib.sha256(source_raw).hexdigest()
    _persist_source_exclusively(root, source_sha, source_raw)

    ledger_path = root / QUOTE_WATERMARK_LEDGER_NAME
    handle = _open_locked_ledger(ledger_path)
    try:
        rows = _read_validate_locked_ledger(handle, ledger_path)
        identity_matches = [
            row
            for row in rows
            if row["pair"] == pair and row["timestamp_utc"] == source["timestamp_utc"]
        ]
        body = {
            "contract": QUOTE_WATERMARK_CONTRACT,
            "sequence": len(rows) + 1,
            "recorded_at_utc": _format_utc(now),
            "timestamp_utc": source["timestamp_utc"],
            "pair": pair,
            "bid": normalized_bid,
            "ask": normalized_ask,
            "source_sha256": source_sha,
            "capture_source_sha256": capture_source_sha256,
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "slippage_pips_per_fill": slippage,
            "financing_pips_per_day": financing,
            "previous_quote_sha256": (
                rows[-1]["quote_sha256"] if rows else GENESIS_QUOTE_SHA256
            ),
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        body["quote_sha256"] = quote_watermark_sha256(body)
        if identity_matches:
            if len(identity_matches) != 1 or identity_matches[0] != body:
                # sequence/recorded_at differ on an exact retry, so compare the
                # immutable quote projection instead of those writer fields.
                existing = identity_matches[0]
                comparable = (
                    "timestamp_utc",
                    "pair",
                    "bid",
                    "ask",
                    "source_sha256",
                    "capture_source_sha256",
                    "acquisition_receipt_sha256",
                    "slippage_pips_per_fill",
                    "financing_pips_per_day",
                )
                if len(identity_matches) != 1 or any(
                    existing[key] != body[key] for key in comparable
                ):
                    raise AiInventoryQuoteWatermarkConflictError(
                        "pair/timestamp quote watermark conflicts with durable truth"
                    )
            return dict(identity_matches[0])
        raw = _canonical_json_bytes(body) + b"\n"
        if len(raw) > MAX_LINE_BYTES:
            raise AiInventoryQuoteWatermarkError("quote watermark row is too large")
        os.lseek(handle.fileno(), 0, os.SEEK_END)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        return body
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_ai_inventory_quote_watermarks(path: Path) -> dict[str, Any]:
    """Validate one complete quote-watermark chain without changing it."""

    handle = _open_existing_locked(path)
    try:
        rows = _read_validate_locked_ledger(handle, path)
        return {
            "valid": True,
            "row_count": len(rows),
            "terminal_quote_sha256": (
                rows[-1]["quote_sha256"] if rows else GENESIS_QUOTE_SHA256
            ),
        }
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def quote_watermark_sha256(value: Mapping[str, Any]) -> str:
    snapshot = dict(value)
    snapshot.pop("quote_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(snapshot)).hexdigest()


def _persist_source_exclusively(root: Path, digest: str, raw: bytes) -> None:
    directory = root / QUOTE_SOURCE_DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise AiInventoryQuoteWatermarkError("quote source directory is unsafe")
    path = directory / f"{digest}.json"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_regular_nofollow(path, MAX_SOURCE_BYTES) != raw:
            raise AiInventoryQuoteWatermarkConflictError(
                "content-addressed quote source has conflicting bytes"
            )
        return
    except OSError as exc:
        raise AiInventoryQuoteWatermarkError("quote source cannot be created") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _open_locked_ledger(path: Path) -> Any:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        create_flags = flags | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(path, create_flags, 0o600)
        except FileExistsError:
            descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryQuoteWatermarkError("quote ledger cannot be opened") from exc
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_LEDGER_BYTES:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise AiInventoryQuoteWatermarkError("quote ledger is not a bounded file")
    return handle


def _open_existing_locked(path: Path) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryQuoteWatermarkError("quote ledger cannot be opened") from exc
    handle = os.fdopen(descriptor, "rb", buffering=0)
    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
    return handle


def _read_validate_locked_ledger(handle: Any, ledger_path: Path) -> list[dict[str, Any]]:
    os.lseek(handle.fileno(), 0, os.SEEK_SET)
    raw = handle.read(MAX_LEDGER_BYTES + 1)
    if len(raw) > MAX_LEDGER_BYTES or (raw and not raw.endswith(b"\n")):
        raise AiInventoryQuoteWatermarkError("quote ledger is oversized or truncated")
    lines = raw.splitlines()
    if len(lines) > MAX_LEDGER_ROWS:
        raise AiInventoryQuoteWatermarkError("quote ledger exceeds row limit")
    rows: list[dict[str, Any]] = []
    previous = GENESIS_QUOTE_SHA256
    previous_timestamp: datetime | None = None
    for index, line in enumerate(lines, 1):
        if not line or len(line) > MAX_LINE_BYTES:
            raise AiInventoryQuoteWatermarkError(f"invalid quote row size at {index}")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AiInventoryQuoteWatermarkError(
                f"invalid quote JSON at row {index}"
            ) from exc
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise AiInventoryQuoteWatermarkError(
                f"quote schema mismatch at row {index}"
            )
        timestamp = _parse_utc(row.get("timestamp_utc"), "quote timestamp")
        recorded = _parse_utc(row.get("recorded_at_utc"), "recorded_at_utc")
        if (
            row.get("contract") != QUOTE_WATERMARK_CONTRACT
            or row.get("sequence") != index
            or row.get("previous_quote_sha256") != previous
            or row.get("quote_sha256") != quote_watermark_sha256(row)
            or row.get("paper_only") is not True
            or row.get("order_authority") != "NONE"
            or row.get("live_permission") is not False
            or not isinstance(row.get("pair"), str)
            or _PAIR_RE.fullmatch(row["pair"]) is None
            or not _SHA256_RE.fullmatch(str(row.get("source_sha256")))
            or not _SHA256_RE.fullmatch(
                str(row.get("acquisition_receipt_sha256"))
            )
            or not _SHA256_RE.fullmatch(
                str(row.get("capture_source_sha256"))
            )
            or row.get("acquisition_receipt_sha256")
            == GENESIS_QUOTE_SHA256
            or row.get("capture_source_sha256") == GENESIS_QUOTE_SHA256
            or recorded < timestamp
            or (previous_timestamp is not None and timestamp < previous_timestamp)
        ):
            raise AiInventoryQuoteWatermarkError(
                f"quote integrity mismatch at row {index}"
            )
        bid = _finite_nonnegative(row.get("bid"), "bid", positive=True)
        ask = _finite_nonnegative(row.get("ask"), "ask", positive=True)
        _finite_nonnegative(
            row.get("slippage_pips_per_fill"), "slippage_pips_per_fill"
        )
        _finite_nonnegative(
            row.get("financing_pips_per_day"), "financing_pips_per_day"
        )
        if ask < bid:
            raise AiInventoryQuoteWatermarkError(f"ask below bid at row {index}")
        source_path = (
            ledger_path.parent
            / QUOTE_SOURCE_DIRECTORY
            / f"{row['source_sha256']}.json"
        )
        source_raw = _read_regular_nofollow(source_path, MAX_SOURCE_BYTES)
        if hashlib.sha256(source_raw).hexdigest() != row["source_sha256"]:
            raise AiInventoryQuoteWatermarkError(
                f"quote source digest mismatch at row {index}"
            )
        try:
            source = json.loads(
                source_raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AiInventoryQuoteWatermarkError(
                f"quote source JSON mismatch at row {index}"
            ) from exc
        expected_source = {
            "contract": QUOTE_SOURCE_CONTRACT,
            "timestamp_utc": row["timestamp_utc"],
            "pair": row["pair"],
            "bid": row["bid"],
            "ask": row["ask"],
            "capture_source_sha256": row["capture_source_sha256"],
            "acquisition_receipt_sha256": row["acquisition_receipt_sha256"],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        if source != expected_source:
            raise AiInventoryQuoteWatermarkError(
                f"quote source content mismatch at row {index}"
            )
        rows.append(row)
        previous = row["quote_sha256"]
        previous_timestamp = timestamp
    return rows


def _require_canonical_room_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AiInventoryQuoteWatermarkError("room_root must be an absolute Path")
    repository_root = _trusted_repository_root()
    rooms_root = canonical_paper_ai_rooms_root(repository_root).resolve(strict=True)
    try:
        root = value.resolve(strict=True)
        relative = root.relative_to(rooms_root)
    except (OSError, ValueError) as exc:
        raise AiInventoryQuoteWatermarkError(
            "room_root is outside the canonical paper-AI rooms root"
        ) from exc
    if (
        root != value
        or len(relative.parts) != 2
        or not all(part.startswith("paper-ai-inventory-") for part in relative.parts)
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise AiInventoryQuoteWatermarkError(
            "room_root is not a canonical paper-AI inventory room"
        )
    return root


def _trusted_repository_root() -> Path:
    try:
        return Path(__file__).resolve(strict=True).parents[2].resolve(strict=True)
    except (IndexError, OSError) as exc:
        raise AiInventoryQuoteWatermarkError(
            "package-derived repository root is unavailable"
        ) from exc


def _read_regular_nofollow(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AiInventoryQuoteWatermarkError("immutable quote source is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise AiInventoryQuoteWatermarkError(
                "immutable quote source is not a bounded regular file"
            )
        raw = os.read(descriptor, maximum_bytes + 1)
        if len(raw) != info.st_size:
            raise AiInventoryQuoteWatermarkError(
                "immutable quote source changed during read"
            )
        return raw
    finally:
        os.close(descriptor)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AiInventoryQuoteWatermarkError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AiInventoryQuoteWatermarkError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise AiInventoryQuoteWatermarkError(f"{label} is not UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_market_open(value: datetime) -> None:
    try:
        is_open = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AiInventoryQuoteWatermarkError(
            "FX market status is unavailable; quote append failed closed"
        ) from exc
    if not is_open:
        raise AiInventoryQuoteWatermarkMarketClosedError(
            "quote watermarks are disabled while FX is closed"
        )


def _finite_nonnegative(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise AiInventoryQuoteWatermarkError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise AiInventoryQuoteWatermarkError(f"{label} is invalid")
    return number


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
