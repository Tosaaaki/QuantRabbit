#!/usr/bin/env python3
"""Fetch provenance-bound S5 bid/ask truth for RANGE scout replay only.

This read-only acquisition path is intentionally separate from
``oanda_history_fetch.py`` because frozen forward-holdout locks bind that
script's exact digest.  It performs no broker writes.  A successful HTTP
response becomes publishable only when its OANDA metadata and every complete
S5 bid/ask candle validate exactly; an empty candle list is a legitimate
no-tick response.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
# Remove every existing occurrence before prepending the canonical repository
# roots.  Merely checking membership would leave an attacker-controlled
# PYTHONPATH entry ahead of a canonical path that appeared later in sys.path.
for path in (SRC_ROOT, SCRIPT_ROOT):
    path_text = str(path)
    sys.path[:] = [item for item in sys.path if item != path_text]
    sys.path.insert(0, path_text)

import oanda_history_fetch as base_fetch  # noqa: E402
import quant_rabbit.broker.oanda as oanda_module  # noqa: E402
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS_ARG  # noqa: E402


_EXPECTED_BASE_FETCH_PATH = SCRIPT_ROOT / "oanda_history_fetch.py"
_EXPECTED_OANDA_MODULE_PATH = SRC_ROOT / "quant_rabbit/broker/oanda.py"
if (
    Path(base_fetch.__file__).resolve() != _EXPECTED_BASE_FETCH_PATH
    or Path(oanda_module.__file__).resolve() != _EXPECTED_OANDA_MODULE_PATH
    or base_fetch.OandaReadOnlyClient is not oanda_module.OandaReadOnlyClient
):
    raise RuntimeError("RANGE truth acquisition imported a non-canonical dependency")
OandaReadOnlyClient = oanda_module.OandaReadOnlyClient
_CANONICAL_OANDA_GET_JSON = OandaReadOnlyClient.get_json


DEFAULT_OUTPUT_DIR = Path("logs/replay/oanda_range_scout_truth")
RANGE_FETCH_SUMMARY_SCHEMA = "QR_OANDA_RANGE_SCOUT_TRUTH_SUMMARY_V1"
RANGE_FETCH_TASK_SCHEMA = "QR_OANDA_RANGE_SCOUT_TRUTH_TASK_V1"
RANGE_TRUTH_RECEIPT_FILE = "range_truth_acquisition_receipts.jsonl"
RANGE_TRUTH_RECEIPT_SCHEMA = "QR_OANDA_RANGE_SCOUT_TRUTH_RECEIPT_V1"
# OANDA v20 documents a maximum of 5,000 candles per request.  This is a
# broker API transport bound, not a market or trading threshold.
OANDA_MAX_CANDLES_PER_REQUEST = 5_000
PRODUCTION_OANDA_BASE_URL = "https://api-fxtrade.oanda.com"
RANGE_TRUTH_RECEIPT_KEYS = {
    "schema_version",
    "sequence",
    "recorded_at_utc",
    "output_root",
    "candle_path",
    "candle_sha256",
    "pair",
    "granularity",
    "price_component",
    "source_base_url",
    "window",
    "rows",
    "task_manifest_sha256",
    "fetch_script_path",
    "fetch_script_sha256",
    "dependencies",
    "previous_receipt_sha256",
    "receipt_sha256",
}
_DEPENDENCY_KEYS = {"role", "path", "sha256"}
_TASK_MANIFEST_KEYS = {
    "schema_version",
    "pair",
    "granularity",
    "price",
    "from",
    "to",
    "path",
    "partial_path",
    "published",
    "windows",
    "requests",
    "rows",
    "errors",
    "dry_run",
    "compressed",
    "max_candles_per_request",
    "include_incomplete",
}
_CANONICAL_FX_PAIR = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z")

FetchTask = base_fetch.FetchTask


@dataclass(frozen=True)
class AcquisitionCodeSnapshot:
    fetch_script_path: Path
    fetch_script_sha256: str
    dependencies: tuple[tuple[str, Path, str], ...]


class AcquisitionCodeDriftError(RuntimeError):
    pass


def main() -> int:
    args = _parse_args()
    pairs = base_fetch._parse_csv(args.pairs)
    if len(set(pairs)) != len(pairs):
        raise ValueError("RANGE truth pairs must be unique")
    for pair in pairs:
        _require_canonical_fx_pair(pair)
    start, end = _resolve_acquisition_window(args)
    list(
        _iter_time_chunks(
            start,
            end,
            granularity="S5",
            max_candles_per_request=args.max_candles_per_request,
        )
    )
    # Resolve credentials and enforce the production endpoint before creating
    # a run directory.  A local configuration failure must not leave an empty
    # acquisition artifact that can be mistaken for an interrupted fetch.
    client = OandaReadOnlyClient()
    output_root = args.output_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": RANGE_FETCH_SUMMARY_SCHEMA,
        "generated_at_utc": base_fetch._iso(datetime.now(timezone.utc)),
        "window": {"from": base_fetch._iso(start), "to": base_fetch._iso(end)},
        "pairs": pairs,
        "granularities": ["S5"],
        "price": "BA",
        "max_candles_per_request": args.max_candles_per_request,
        "output_root": str(output_root),
        "output_dir": str(run_dir),
        "tasks": [],
        "errors": [],
        "total_rows": 0,
        "total_requests": 0,
        "dry_run": bool(args.dry_run),
        "include_incomplete": False,
    }
    for pair in pairs:
        task = FetchTask(
            pair=pair,
            granularity="S5",
            start=start,
            end=end,
            price="BA",
        )
        task_summary = _fetch_task(
            client,
            task,
            run_dir=run_dir,
            receipt_root=output_root,
            max_candles_per_request=args.max_candles_per_request,
            sleep_seconds=args.sleep_seconds,
            retries=args.retries,
            compress=args.compress,
            dry_run=args.dry_run,
        )
        summary["tasks"].append(task_summary)
        summary["total_rows"] += int(task_summary.get("rows", 0))
        summary["total_requests"] += int(task_summary.get("requests", 0))
        summary["errors"].extend(task_summary.get("errors", []))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not args.dry_run and not summary["errors"]:
        latest_path = output_root / "latest_summary.json"
        latest_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(
        f"rows={summary['total_rows']} requests={summary['total_requests']} "
        f"errors={len(summary['errors'])}"
    )
    return 1 if summary["errors"] else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    parser.add_argument("--from", dest="time_from")
    parser.add_argument("--to", dest="time_to")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-candles-per-request", type=int, default=4500)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument("--compress", dest="compress", action="store_true", default=True)
    compression.add_argument("--no-compress", dest="compress", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_acquisition_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    start, end = base_fetch._resolve_window(args)
    # Only fully closed S5 intervals are receipt-eligible.  In particular, a
    # default --to=now request must stop at the latest completed S5 boundary so
    # OANDA's current incomplete candle cannot poison every standard run.
    start = _floor_s5_boundary(start)
    end = _floor_s5_boundary(end)
    if start >= end:
        raise ValueError("canonical S5 acquisition window is empty")
    return start, end


def _floor_s5_boundary(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("S5 acquisition boundary must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    epoch = math.floor(utc.timestamp() / 5.0) * 5
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _fetch_task(
    client: Any,
    task: FetchTask,
    *,
    run_dir: Path,
    receipt_root: Path,
    max_candles_per_request: int,
    sleep_seconds: float,
    retries: int,
    compress: bool = True,
    dry_run: bool,
) -> dict[str, Any]:
    _require_canonical_oanda_client(client)
    _require_canonical_fx_pair(task.pair)
    if task.granularity != "S5" or task.price != "BA":
        raise ValueError("RANGE scout truth acquisition is fixed to S5/BA")
    if (
        task.start.tzinfo is None
        or task.end.tzinfo is None
        or task.start >= task.end
    ):
        raise ValueError("RANGE scout truth window must be positive and timezone-aware")
    if any(
        boundary.microsecond != 0 or boundary.timestamp() % 5 != 0
        for boundary in (task.start, task.end)
    ):
        raise ValueError("RANGE scout truth window must use exact S5 boundaries")
    if task.end > datetime.now(timezone.utc):
        raise ValueError("RANGE scout truth window must be mature before acquisition")
    list(
        _iter_time_chunks(
            task.start,
            task.end,
            granularity="S5",
            max_candles_per_request=max_candles_per_request,
        )
    )
    validated_client = _ValidatedCandleClient(client, task=task)
    task_summary = base_fetch._fetch_task(
        validated_client,
        task,
        run_dir=run_dir,
        receipt_root=None,
        max_candles_per_request=max_candles_per_request,
        sleep_seconds=sleep_seconds,
        retries=retries,
        include_incomplete=False,
        compress=compress,
        dry_run=dry_run,
    )
    task_summary = {
        "schema_version": RANGE_FETCH_TASK_SCHEMA,
        **task_summary,
        "max_candles_per_request": max_candles_per_request,
        "include_incomplete": False,
    }
    if task_summary.get("published") is True:
        if not _code_snapshot_unchanged(_MODULE_CODE_SNAPSHOT):
            return _invalidate_published_task_for_code_drift(task_summary)
        try:
            receipt = _append_range_truth_receipt(
                output_root=receipt_root,
                task=task,
                candle_path=Path(str(task_summary["path"])),
                rows=int(task_summary["rows"]),
                task_manifest_sha256=task_manifest_sha256(task_summary),
                source_base_url=client.base_url,
                code_snapshot=_MODULE_CODE_SNAPSHOT,
            )
        except AcquisitionCodeDriftError:
            return _invalidate_published_task_for_code_drift(task_summary)
        task_summary["range_truth_acquisition_receipt_sha256"] = receipt[
            "receipt_sha256"
        ]
    return task_summary


def _invalidate_published_task_for_code_drift(
    task_summary: dict[str, Any],
) -> dict[str, Any]:
    published = Path(str(task_summary["path"]))
    partial = published.with_name(f"{published.name}.partial")
    if published.is_file():
        published.replace(partial)
    task_summary["published"] = False
    task_summary["partial_path"] = str(partial)
    task_summary.setdefault("errors", []).append(
        {"error": "acquisition_code_or_dependency_sha_drift"}
    )
    task_summary.pop("range_truth_acquisition_receipt_sha256", None)
    return task_summary


class _ValidatedCandleClient:
    def __init__(self, delegate: OandaReadOnlyClient, *, task: FetchTask) -> None:
        self._delegate = delegate
        self._task = task
        self._seen_candles: dict[datetime, str] = {}

    def get_json(self, path: str, query: dict[str, Any]) -> dict[str, Any]:
        expected_path = f"/v3/instruments/{self._task.pair}/candles"
        if path != expected_path or set(query) != {
            "granularity",
            "from",
            "to",
            "price",
            "includeFirst",
        }:
            raise ValueError("unexpected RANGE truth request shape")
        if (
            query.get("granularity") != "S5"
            or query.get("price") != "BA"
            or query.get("includeFirst") != "true"
        ):
            raise ValueError("unexpected RANGE truth request parameters")
        window_start = base_fetch._parse_time(str(query["from"]))
        window_end = base_fetch._parse_time(str(query["to"]))
        if self._delegate.base_url != PRODUCTION_OANDA_BASE_URL:
            raise ValueError("RANGE truth source endpoint changed during acquisition")
        # Call the import-time canonical implementation directly.  An
        # instance-level get_json replacement must never become a receipt-
        # eligible truth source.
        payload = _CANONICAL_OANDA_GET_JSON(self._delegate, path, query)
        if not _valid_candle_payload(
            payload,
            task=self._task,
            window_start=window_start,
            window_end=window_end,
        ):
            raise ValueError("OANDA RANGE truth payload metadata or candle shape invalid")
        for candle in payload["candles"]:
            candle_time = base_fetch._parse_time(str(candle["time"]))
            canonical = json.dumps(
                candle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if candle_time in self._seen_candles:
                prior = self._seen_candles[candle_time]
                kind = "conflicting" if prior != canonical else "duplicate"
                raise ValueError(f"{kind} RANGE truth candle across request windows")
            self._seen_candles[candle_time] = canonical
        return payload


def _valid_candle_payload(
    payload: object,
    *,
    task: FetchTask,
    window_start: datetime,
    window_end: datetime,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("instrument") != task.pair or payload.get("granularity") != "S5":
        return False
    candles = payload.get("candles")
    if not isinstance(candles, list):
        return False
    previous_time: datetime | None = None
    for candle in candles:
        if not isinstance(candle, dict) or set(candle) != {
            "time",
            "complete",
            "volume",
            "bid",
            "ask",
        }:
            return False
        # A returned incomplete bar is not a natural no-tick absence.  It
        # invalidates the task so the base writer produces only a partial file
        # and this wrapper issues no receipt.
        if candle.get("complete") is not True:
            return False
        volume = candle.get("volume")
        if not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
            return False
        timestamp = candle.get("time")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            return False
        try:
            parsed_time = base_fetch._parse_time(timestamp)
        except (TypeError, ValueError):
            return False
        if (
            not window_start <= parsed_time < window_end
            or parsed_time.microsecond != 0
            or parsed_time.timestamp() % 5 != 0
        ):
            return False
        if previous_time is not None and parsed_time <= previous_time:
            return False
        previous_time = parsed_time
        if not _valid_ohlc(candle.get("bid")) or not _valid_ohlc(candle.get("ask")):
            return False
        if any(
            float(candle["ask"][field]) <= float(candle["bid"][field])
            for field in ("o", "h", "l", "c")
        ):
            return False
    return True


def _require_canonical_oanda_client(client: object) -> None:
    if (
        type(client) is not OandaReadOnlyClient
        or OandaReadOnlyClient.get_json is not _CANONICAL_OANDA_GET_JSON
        or "get_json" in vars(client)
        or getattr(client.get_json, "__func__", None) is not _CANONICAL_OANDA_GET_JSON
        or client.base_url != PRODUCTION_OANDA_BASE_URL
    ):
        raise ValueError(
            "receipt-eligible RANGE truth requires the canonical production "
            "OANDA read client"
        )


def _require_canonical_fx_pair(pair: object) -> None:
    if not isinstance(pair, str) or _CANONICAL_FX_PAIR.fullmatch(pair) is None:
        raise ValueError(
            "RANGE truth acquisition pair must be a canonical FX instrument "
            "such as CAD_JPY"
        )


def _valid_ohlc(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"o", "h", "l", "c"}:
        return False
    parsed: dict[str, float] = {}
    for key in ("o", "h", "l", "c"):
        raw = value.get(key)
        if isinstance(raw, bool):
            return False
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number) or number <= 0.0:
            return False
        parsed[key] = number
    return (
        parsed["l"] <= parsed["o"] <= parsed["h"]
        and parsed["l"] <= parsed["c"] <= parsed["h"]
    )


def _iter_time_chunks(
    start: datetime,
    end: datetime,
    *,
    granularity: str,
    max_candles_per_request: int,
) -> Iterable[tuple[datetime, datetime]]:
    if granularity != "S5":
        raise ValueError("RANGE scout truth acquisition is fixed to S5")
    if (
        not isinstance(max_candles_per_request, int)
        or isinstance(max_candles_per_request, bool)
        or not 1 <= max_candles_per_request <= OANDA_MAX_CANDLES_PER_REQUEST
    ):
        raise ValueError(
            "max_candles_per_request must be an integer in "
            f"1..{OANDA_MAX_CANDLES_PER_REQUEST}"
        )
    yield from base_fetch._iter_time_chunks(
        start,
        end,
        granularity=granularity,
        max_candles_per_request=max_candles_per_request,
    )


def expected_dependency_records() -> list[dict[str, str]]:
    dependencies = (
        ("BASE_HISTORY_FETCH", Path(base_fetch.__file__).resolve()),
        ("OANDA_READ_CLIENT", Path(oanda_module.__file__).resolve()),
    )
    return [
        {"role": role, "path": str(path), "sha256": _sha256_path(path)}
        for role, path in dependencies
    ]


def _capture_code_snapshot() -> AcquisitionCodeSnapshot:
    fetch_script = Path(__file__).resolve()
    dependencies = tuple(
        (record["role"], Path(record["path"]), record["sha256"])
        for record in expected_dependency_records()
    )
    return AcquisitionCodeSnapshot(
        fetch_script_path=fetch_script,
        fetch_script_sha256=_sha256_path(fetch_script),
        dependencies=dependencies,
    )


def _code_snapshot_unchanged(snapshot: AcquisitionCodeSnapshot) -> bool:
    try:
        if _sha256_path(snapshot.fetch_script_path) != snapshot.fetch_script_sha256:
            return False
        return all(
            _sha256_path(path) == expected_sha
            for _role, path, expected_sha in snapshot.dependencies
        )
    except OSError:
        return False


def _snapshot_dependency_records(
    snapshot: AcquisitionCodeSnapshot,
) -> list[dict[str, str]]:
    return [
        {"role": role, "path": str(path), "sha256": sha256}
        for role, path, sha256 in snapshot.dependencies
    ]


def task_manifest_sha256(task_summary: dict[str, Any]) -> str:
    if not _TASK_MANIFEST_KEYS.issubset(task_summary):
        raise ValueError("RANGE truth task manifest fields missing")
    body = {key: task_summary[key] for key in sorted(_TASK_MANIFEST_KEYS)}
    return _content_sha256(body)


def _append_range_truth_receipt(
    *,
    output_root: Path,
    task: FetchTask,
    candle_path: Path,
    rows: int,
    task_manifest_sha256: str,
    source_base_url: str,
    code_snapshot: AcquisitionCodeSnapshot,
) -> dict[str, Any]:
    root = output_root.resolve()
    published = candle_path.resolve(strict=True)
    try:
        published.relative_to(root)
    except ValueError as exc:
        raise ValueError("published RANGE truth file must remain inside output root") from exc
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("published RANGE truth row count must be nonnegative")
    if source_base_url != PRODUCTION_OANDA_BASE_URL:
        raise ValueError("published RANGE truth source endpoint is not production OANDA")
    receipt_path = root / RANGE_TRUTH_RECEIPT_FILE
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if not _code_snapshot_unchanged(code_snapshot):
            raise AcquisitionCodeDriftError(
                "acquisition code changed before receipt commit"
            )
        handle.seek(0)
        prior = _validate_range_truth_receipt_chain(handle.read().encode("utf-8"))
        previous_sha = str(prior[-1]["receipt_sha256"]) if prior else None
        body: dict[str, Any] = {
            "schema_version": RANGE_TRUTH_RECEIPT_SCHEMA,
            "sequence": len(prior) + 1,
            "recorded_at_utc": base_fetch._iso(datetime.now(timezone.utc)),
            "output_root": str(root),
            "candle_path": str(published),
            "candle_sha256": _sha256_path(published),
            "pair": task.pair,
            "granularity": task.granularity,
            "price_component": task.price,
            "source_base_url": source_base_url,
            "window": {
                "from_utc": base_fetch._iso(task.start),
                "to_utc": base_fetch._iso(task.end),
            },
            "rows": rows,
            "task_manifest_sha256": task_manifest_sha256,
            "fetch_script_path": str(code_snapshot.fetch_script_path),
            "fetch_script_sha256": code_snapshot.fetch_script_sha256,
            "dependencies": _snapshot_dependency_records(code_snapshot),
            "previous_receipt_sha256": previous_sha,
        }
        receipt = {**body, "receipt_sha256": _content_sha256(body)}
        handle.seek(0, os.SEEK_END)
        handle.write(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def _validate_range_truth_receipt_chain(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_sha: str | None = None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("RANGE truth receipt ledger is not UTF-8") from exc
    for raw in text.splitlines():
        if not raw.strip():
            raise ValueError("RANGE truth receipt ledger contains a blank row")
        try:
            item = json.loads(raw, object_pairs_hook=_no_duplicate_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("RANGE truth receipt ledger is malformed") from exc
        if not isinstance(item, dict) or set(item) != RANGE_TRUTH_RECEIPT_KEYS:
            raise ValueError("RANGE truth receipt schema invalid")
        if item.get("schema_version") != RANGE_TRUTH_RECEIPT_SCHEMA:
            raise ValueError("RANGE truth receipt version invalid")
        if (
            not isinstance(item.get("sequence"), int)
            or isinstance(item.get("sequence"), bool)
            or item.get("sequence") != len(rows) + 1
        ):
            raise ValueError("RANGE truth receipt sequence invalid")
        if item.get("previous_receipt_sha256") != previous_sha:
            raise ValueError("RANGE truth receipt chain broken")
        dependencies = item.get("dependencies")
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(dep, dict) or set(dep) != _DEPENDENCY_KEYS for dep in dependencies)
        ):
            raise ValueError("RANGE truth dependency schema invalid")
        body = {key: value for key, value in item.items() if key != "receipt_sha256"}
        expected = _content_sha256(body)
        if item.get("receipt_sha256") != expected:
            raise ValueError("RANGE truth receipt digest mismatch")
        previous_sha = expected
        rows.append(item)
    return rows


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _content_sha256(value: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_MODULE_CODE_SNAPSHOT = _capture_code_snapshot()


if __name__ == "__main__":
    raise SystemExit(main())
