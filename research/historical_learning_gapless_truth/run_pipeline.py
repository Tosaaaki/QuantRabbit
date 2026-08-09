#!/usr/bin/env python3
"""Bounded read-only historical tick acquisition and causal admission replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import http.client
import importlib.util
import json
import lzma
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
EPISODES = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
EPISODE_SHA256 = "efcf6b0fb675050d6a08efc0119065e0874e50e1c51373a0c0fb61bb6ebd815e"
PREREG = ROOT / "preregister_v2.json"
PAIRS = {"AUD_JPY": "AUDJPY", "EUR_JPY": "EURJPY", "EUR_USD": "EURUSD"}
WINDOW_START = "2026-05-06T07:46:03.151624347Z"
ANCHOR = "2026-07-09T07:46:03.151624347Z"
SOFT_CAP = 2 * 1024**3
HARD_CAP = 5 * 1024**3
RECORD = struct.Struct(">3i2f")
NY = ZoneInfo("America/New_York")
SOURCE_BASE = "https://datafeed.dukascopy.com/datafeed"


@dataclass(frozen=True)
class HourKey:
    pair: str
    start: datetime

    @property
    def symbol(self) -> str:
        return PAIRS[self.pair]

    @property
    def url(self) -> str:
        # Dukascopy datafeed months are zero-based; this is verified by a
        # cross-source spot check against the same UTC hour in OANDA S5.
        return (
            f"{SOURCE_BASE}/{self.symbol}/{self.start.year}/"
            f"{self.start.month - 1:02d}/{self.start.day:02d}/"
            f"{self.start.hour:02d}h_ticks.bi5"
        )

    @property
    def path(self) -> Path:
        return (
            CACHE / "raw" / "dukascopy_tick" / self.pair /
            self.start.strftime("%Y-%m-%d") /
            f"{self.start:%H}h_ticks.bi5"
        )


@dataclass(frozen=True)
class Tick:
    time: datetime
    ask: float
    bid: float
    ask_volume: float
    bid_volume: float


def parse_time(text: str) -> datetime:
    normalized = str(text).replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+([+-])", r"\1\2", normalized)
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            # A downloader atomically renames/unlinks its run-owned temporary
            # file while the bounded-size observer walks the cache.
            continue
    return total


def read_episodes() -> list[dict[str, Any]]:
    if sha256(EPISODES) != EPISODE_SHA256:
        raise RuntimeError("frozen episode input changed")
    rows = [json.loads(line) for line in EPISODES.read_text().splitlines() if line.strip()]
    start, end = parse_time(WINDOW_START), parse_time(ANCHOR)
    return [
        row for row in rows
        if row.get("label_status") == "ACTUAL_AFTER_COST"
        and start <= parse_time(row["feature_at_utc"]) <= end
    ]


def required_hours(episodes: Iterable[dict[str, Any]]) -> list[HourKey]:
    selected: set[HourKey] = set()
    for row in episodes:
        pair = str(row.get("pair"))
        if pair not in PAIRS:
            continue
        feature_at = parse_time(row["feature_at_utc"])
        end = feature_at.replace(minute=0, second=0, microsecond=0)
        # Four hours supply 48 M5 bars. One extra hour supplies a causal quote
        # for explicit no-tick carry at the left boundary.
        cursor = (feature_at - timedelta(hours=4)).replace(minute=0, second=0, microsecond=0)
        while cursor <= end:
            selected.add(HourKey(pair, cursor))
            cursor += timedelta(hours=1)
    return sorted(selected, key=lambda item: (item.pair, item.start))


def market_closed(hour: datetime) -> bool:
    local = hour.astimezone(NY)
    weekday = local.weekday()
    if weekday == 4 and local.hour >= 17:
        return True
    if weekday == 5:
        return True
    return weekday == 6 and local.hour < 17


def curl_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="qr-tick-", suffix=".tmp", dir=destination.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "10", "--max-time", "45", "--retry", "5",
            "--retry-all-errors", "--retry-delay", "5", "--retry-max-time", "180",
            "--output", str(temp), url,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"curl_exit_{completed.returncode}:{completed.stderr.strip()[:200]}")
        if temp.stat().st_size == 0:
            raise RuntimeError("empty_source_file")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


class PersistentHistoricalFeed:
    """One bounded HTTPS connection for courteous serial archive retrieval."""

    def __init__(self, base: str) -> None:
        parsed = urlsplit(base)
        if parsed.scheme != "https" or not parsed.hostname:
            raise RuntimeError("historical feed must use https")
        self.host = parsed.hostname
        self.connection: http.client.HTTPSConnection | None = None

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _connect(self) -> http.client.HTTPSConnection:
        if self.connection is None:
            self.connection = http.client.HTTPSConnection(self.host, timeout=45)
        return self.connection

    def download(self, url: str, destination: Path) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != self.host:
            raise RuntimeError("source host boundary changed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        last_error = "unknown"
        for attempt in range(6):
            temp: Path | None = None
            try:
                connection = self._connect()
                connection.request("GET", parsed.path, headers={
                    "Accept": "application/octet-stream",
                    "User-Agent": "QuantRabbit-read-only-historical-research/1.0",
                })
                response = connection.getresponse()
                if response.status != 200:
                    response.read()
                    last_error = f"http_{response.status}:{response.reason}"
                    self.close()
                    retry_after = response.getheader("Retry-After")
                    pause = int(retry_after) if retry_after and retry_after.isdigit() else min(20, 2**attempt)
                    time.sleep(pause)
                    continue
                with tempfile.NamedTemporaryFile(prefix="qr-tick-", suffix=".tmp", dir=destination.parent, delete=False) as handle:
                    temp = Path(handle.name)
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                if temp.stat().st_size == 0:
                    raise RuntimeError("empty_source_file")
                temp.replace(destination)
                return
            except Exception as exc:  # noqa: BLE001 - bounded retry records the final transport cause.
                last_error = f"{type(exc).__name__}:{exc}"
                self.close()
                time.sleep(min(20, 2**attempt))
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        raise RuntimeError(f"persistent_feed_failed:{last_error}")


def decode_hour(key: HourKey, path: Path) -> tuple[list[Tick], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = lzma.decompress(raw)
    except lzma.LZMAError as exc:
        raise RuntimeError(f"lzma_decode_failed:{exc}") from exc
    if len(payload) % RECORD.size:
        raise RuntimeError(f"record_remainder:{len(payload) % RECORD.size}")
    scale = 1000.0 if key.pair.endswith("_JPY") else 100000.0
    ticks: list[Tick] = []
    last_ms = -1
    duplicate_timestamps = 0
    exact_duplicates = 0
    previous: tuple[int, int, int, float, float] | None = None
    for raw_record in RECORD.iter_unpack(payload):
        millis, ask_i, bid_i, ask_volume, bid_volume = raw_record
        if not 0 <= millis < 3_600_000:
            raise RuntimeError(f"millisecond_out_of_hour:{millis}")
        if millis < last_ms:
            raise RuntimeError(f"non_monotonic_tick:{millis}<{last_ms}")
        if millis == last_ms:
            duplicate_timestamps += 1
        if raw_record == previous:
            exact_duplicates += 1
        previous = raw_record
        last_ms = millis
        if ask_i <= 0 or bid_i <= 0:
            raise RuntimeError(f"non_positive_quote:{ask_i}:{bid_i}")
        if not math.isfinite(ask_volume) or not math.isfinite(bid_volume) or ask_volume < 0 or bid_volume < 0:
            raise RuntimeError(f"invalid_volume:{ask_volume}:{bid_volume}")
        ask, bid = ask_i / scale, bid_i / scale
        if bid > ask:
            raise RuntimeError(f"crossed_quote:{bid}>{ask}")
        ticks.append(Tick(key.start + timedelta(milliseconds=millis), ask, bid, ask_volume, bid_volume))
    return ticks, {
        "rows": len(ticks),
        "duplicate_timestamps": duplicate_timestamps,
        "exact_duplicate_records": exact_duplicates,
        "raw_bytes": len(raw),
        "decoded_bytes": len(payload),
        "first_tick_utc": iso(ticks[0].time) if ticks else None,
        "last_tick_utc": iso(ticks[-1].time) if ticks else None,
    }


def acquire_one(key: HourKey, downloader: Any = curl_download) -> dict[str, Any]:
    path = key.path
    fetched = False
    error = None
    if not path.exists():
        if market_closed(key.start):
            error = "SCHEDULED_MARKET_CLOSED_NO_FETCH"
        elif directory_bytes(CACHE) >= HARD_CAP:
            raise RuntimeError("HARD_CAP_REACHED")
        else:
            try:
                downloader(key.url, path)
                fetched = True
            except Exception as exc:  # noqa: BLE001 - captured into the acquisition ledger.
                error = f"{type(exc).__name__}:{exc}"
    audit: dict[str, Any] = {}
    if error is None:
        try:
            _, audit = decode_hour(key, path)
        except Exception as exc:  # noqa: BLE001 - captured into the acquisition ledger.
            error = f"{type(exc).__name__}:{exc}"
    return {
        "source": "DUKASCOPY_DATAFEED_TICK",
        "pair": key.pair,
        "utc_from": iso(key.start),
        "utc_to_exclusive": iso(key.start + timedelta(hours=1)),
        "granularity": "TICK",
        "price_type": "BID_ASK",
        "timezone": "UTC",
        "path": str(path.relative_to(REPO)),
        "url": key.url,
        "fetched_this_run": fetched,
        "fetched_at_utc": iso(datetime.now(timezone.utc)),
        "complete": error is None,
        "market_closed": market_closed(key.start),
        "gap_reason": None if error is None else ("MARKET_CLOSED" if market_closed(key.start) else "ACQUISITION_DEFECT"),
        "sha256": sha256(path) if path.exists() and error is None else None,
        "error": error,
        **audit,
    }


def duplicate_readback(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    feed = PersistentHistoricalFeed(SOURCE_BASE)
    try:
        for pair in PAIRS:
            candidate = next((row for row in entries if row["pair"] == pair and row["complete"]), None)
            if candidate is None:
                checks.append({"pair": pair, "match": False, "reason": "NO_COMPLETE_CANDIDATE"})
                continue
            original = REPO / candidate["path"]
            with tempfile.TemporaryDirectory(prefix="qr-tick-double-") as tmp:
                copy = Path(tmp) / original.name
                try:
                    feed.download(candidate["url"], copy)
                    second_ticks, second_audit = decode_hour(HourKey(pair, parse_time(candidate["utc_from"])), copy)
                    checks.append({
                        "pair": pair,
                        "utc_from": candidate["utc_from"],
                        "first_sha256": sha256(original),
                        "second_sha256": sha256(copy),
                        "first_bytes": original.stat().st_size,
                        "second_bytes": copy.stat().st_size,
                        "first_rows": candidate["rows"],
                        "second_rows": len(second_ticks),
                        "second_decode_audit": second_audit,
                        "match": original.read_bytes() == copy.read_bytes() and candidate["rows"] == len(second_ticks),
                    })
                except Exception as exc:  # noqa: BLE001 - failed oracle must fail admission, not erase the manifest.
                    checks.append({
                        "pair": pair,
                        "utc_from": candidate["utc_from"],
                        "first_sha256": sha256(original),
                        "match": False,
                        "reason": f"SECOND_FETCH_FAILED:{type(exc).__name__}:{str(exc)[:200]}",
                    })
    finally:
        feed.close()
    return checks


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def acquire(workers: int) -> dict[str, Any]:
    episodes = read_episodes()
    hours = required_hours(episodes)
    before = directory_bytes(CACHE)
    entries: list[dict[str, Any]] = []
    def checkpoint() -> None:
        if len(entries) % 25 == 0:
            write_json(CACHE / "partial_manifest_v2.json", {
                "contract": "historical_tick_cache_partial_v2",
                "generated_at_utc": iso(datetime.now(timezone.utc)),
                "required_hours": len(hours),
                "completed_futures": len(entries),
                "complete_files": sum(row["complete"] for row in entries),
                "market_open_defects": sum(not row["complete"] and not row["market_closed"] for row in entries),
                "cache_bytes": directory_bytes(CACHE),
                "entries": sorted(entries, key=lambda row: (row["pair"], row["utc_from"])),
            })

    def enforce_caps() -> None:
        size = directory_bytes(CACHE)
        if size >= HARD_CAP:
            raise RuntimeError("HARD_CAP_REACHED")
        if size >= SOFT_CAP:
            raise RuntimeError("SOFT_CAP_CHECKPOINT_REACHED")

    if workers == 1:
        feed = PersistentHistoricalFeed(SOURCE_BASE)
        try:
            for key in hours:
                entries.append(acquire_one(key, feed.download))
                checkpoint()
                enforce_caps()
                time.sleep(0.1)
        finally:
            feed.close()
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
            futures = {pool.submit(acquire_one, key): key for key in hours}
            for future in as_completed(futures):
                entries.append(future.result())
                checkpoint()
                enforce_caps()
    entries.sort(key=lambda row: (row["pair"], row["utc_from"]))
    manifest = {
        "contract": "historical_tick_cache_manifest_v2",
        "generated_at_utc": iso(datetime.now(timezone.utc)),
        "source": {
            "name": "Dukascopy official historical tick datafeed",
            "historical_export": "https://www.dukascopy.com/swiss/english/marketwatch/historical/",
            "tick_documentation": "https://www.dukascopy.com/wiki/en/development/strategy-api/historical-data/history-ticks/",
            "tester_documentation": "https://www.dukascopy.com/wiki/en/forex-cfds/jforex/historical-tester/",
            "license_disclaimer": "https://www.dukascopy.com/swiss/english/legal-pages/important-disclaimer/",
            "evidence_grade": "PARTIAL_REPRODUCIBLE_OFFICIAL_SOURCE_ENDPOINT_SCHEMA_LOCALLY_ORACLED",
            "boundary": "feature source only; never an OANDA fill substitute",
        },
        "storage": {"before_bytes": before, "after_bytes": directory_bytes(CACHE), "soft_cap_bytes": SOFT_CAP, "hard_cap_bytes": HARD_CAP},
        "episode_scope": {"all_labeled": len(episodes), "allowed_pair_labeled": sum(row["pair"] in PAIRS for row in episodes), "pair_counts": dict(Counter(row["pair"] for row in episodes))},
        "required_hours": len(hours),
        "entries": entries,
        "double_download_checks": duplicate_readback(entries),
    }
    manifest["admission"] = {
        "all_required_hours_complete": all(row["complete"] or row["market_closed"] for row in entries),
        "market_open_defects": sum(not row["complete"] and not row["market_closed"] for row in entries),
        "market_open_defect_free": not any(not row["complete"] and not row["market_closed"] for row in entries),
        "bid_ask_valid": all(row["error"] is None or row["market_closed"] for row in entries),
        "double_download_match": all(row["match"] for row in manifest["double_download_checks"]),
    }
    write_json(CACHE / "manifest_v2.json", manifest)
    (CACHE / "partial_manifest_v2.json").unlink(missing_ok=True)
    return manifest


def floor_s5(value: datetime) -> datetime:
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 5, tz=timezone.utc)


def ohlc(values: list[float]) -> dict[str, float]:
    return {"o": values[0], "h": max(values), "l": min(values), "c": values[-1]}


def build_s5() -> dict[str, Any]:
    manifest = json.loads((CACHE / "manifest_v2.json").read_text())
    entries = {(row["pair"], row["utc_from"]): row for row in manifest["entries"]}
    source_reports: dict[str, Any] = {}
    for pair in PAIRS:
        pair_entries = [row for row in manifest["entries"] if row["pair"] == pair]
        pair_entries.sort(key=lambda row: row["utc_from"])
        output_dir = CACHE / "derived" / "dukascopy_s5" / pair
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{pair}_S5_BA_REQUIRED_64D.jsonl.gz"
        lineage = output_dir / f"{pair}_M5_LINEAGE_REQUIRED_64D.jsonl.gz"
        rows = observed = carried = defects = 0
        m5_counts: dict[datetime, Counter[str]] = defaultdict(Counter)
        previous_tick: Tick | None = None
        previous_hour: datetime | None = None
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            for entry in pair_entries:
                hour = parse_time(entry["utc_from"])
                contiguous = previous_hour is not None and hour - previous_hour == timedelta(hours=1)
                if not contiguous:
                    previous_tick = None
                previous_hour = hour
                if not entry["complete"]:
                    defects += int(not entry["market_closed"])
                    continue
                key = HourKey(pair, hour)
                ticks, _ = decode_hour(key, REPO / entry["path"])
                buckets: dict[datetime, list[Tick]] = defaultdict(list)
                for tick in ticks:
                    buckets[floor_s5(tick.time)].append(tick)
                cursor = hour
                for _ in range(720):
                    bucket_ticks = buckets.get(cursor, [])
                    lineage_code: str
                    if bucket_ticks:
                        bids = [tick.bid for tick in bucket_ticks]
                        asks = [tick.ask for tick in bucket_ticks]
                        previous_tick = bucket_ticks[-1]
                        lineage_code = "OBSERVED_TICKS"
                        observed += 1
                    elif previous_tick is not None:
                        bids = [previous_tick.bid]
                        asks = [previous_tick.ask]
                        lineage_code = "CARRY_RAW_NO_TRADE"
                        carried += 1
                    else:
                        cursor += timedelta(seconds=5)
                        continue
                    row = {
                        "time": iso(cursor), "pair": pair, "granularity": "S5", "price": "BA",
                        "complete": True, "volume": len(bucket_ticks), "bid": ohlc(bids), "ask": ohlc(asks),
                        "source": "DUKASCOPY_DATAFEED_TICK", "source_lineage": lineage_code,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    rows += 1
                    m5_start = cursor.replace(minute=cursor.minute - cursor.minute % 5, second=0, microsecond=0)
                    m5_counts[m5_start][lineage_code] += 1
                    cursor += timedelta(seconds=5)
        with gzip.open(lineage, "wt", encoding="utf-8") as handle:
            for start, counts in sorted(m5_counts.items()):
                total = sum(counts.values())
                handle.write(json.dumps({
                    "pair": pair, "m5_start_utc": iso(start), "s5_total": total,
                    "observed_s5": counts["OBSERVED_TICKS"], "carry_no_trade_s5": counts["CARRY_RAW_NO_TRADE"],
                    "complete": total == 60,
                }, sort_keys=True) + "\n")
        source_reports[pair] = {
            "s5_path": str(output.relative_to(REPO)), "s5_sha256": sha256(output), "s5_rows": rows,
            "observed_s5": observed, "carry_no_trade_s5": carried, "market_open_defect_hours": defects,
            "m5_lineage_path": str(lineage.relative_to(REPO)), "m5_lineage_sha256": sha256(lineage),
            "complete_m5": sum(sum(c.values()) == 60 for c in m5_counts.values()),
        }
    report = {"contract": "dukascopy_tick_to_s5_build_v2", "generated_at_utc": iso(datetime.now(timezone.utc)), "sources": source_reports}
    write_json(CACHE / "build_report_v2.json", report)
    return report


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def episode_lineage(build: dict[str, Any]) -> dict[str, Any]:
    parent = import_module("gapless_lineage_parent", REPO / "research/historical_learning_price_action_admission/run_price_action_admission.py")
    admission = import_module("gapless_admission_parent", REPO / "research/historical_learning_admission/run_admission.py")
    episodes = read_episodes()
    pair_bars: dict[str, list[Any]] = {}
    for pair, source in build["sources"].items():
        bars, _ = parent.load_bars(REPO / source["s5_path"], pair, admission.parse_time)
        pair_bars[pair] = bars
    rows, reasons = parent.attach_features(episodes, pair_bars, admission.parse_time)
    ledger = []
    for row in rows:
        available = row.get("price_action_features") is not None
        pair = row["pair"]
        feature_at = parse_time(row["feature_at_utc"])
        start_hour = (feature_at - timedelta(hours=4)).replace(minute=0, second=0, microsecond=0)
        end_hour = feature_at.replace(minute=0, second=0, microsecond=0)
        lookback_crosses_market_close = any(
            market_closed(start_hour + timedelta(hours=offset))
            for offset in range(int((end_hour - start_hour).total_seconds() // 3600) + 1)
        )
        if available:
            coverage_reason = None
        elif pair not in PAIRS:
            coverage_reason = "PAIR_OUT_OF_SCOPE_BY_EXPLICIT_ALLOWLIST"
        elif lookback_crosses_market_close:
            coverage_reason = "MARKET_CLOSED_LOOKBACK_EXCLUDED"
        else:
            coverage_reason = "GAP_OR_LOOKBACK_INCOMPLETE"
        ledger.append({
            "episode_id": row["episode_id"], "pair": pair, "feature_at_utc": row["feature_at_utc"],
            "strict_48_m5_available": available,
            "coverage_reason": coverage_reason,
            "lookback_crosses_market_close": lookback_crosses_market_close,
            "feature_source": "DUKASCOPY_DATAFEED_TICK" if pair in PAIRS else None,
            "execution_source": "OANDA_ACTUAL_AFTER_COST",
            "source_boundary_preserved": True,
        })
    path = ROOT / "episode_coverage_v2.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger))
    eligible = [row for row in ledger if row["pair"] in PAIRS]
    report = {
        "contract": "episode_gap_coverage_audit_v2", "episodes_all": len(ledger), "episodes_allowed_pairs": len(eligible),
        "strict_available_all": sum(row["strict_48_m5_available"] for row in ledger),
        "strict_available_allowed_pairs": sum(row["strict_48_m5_available"] for row in eligible),
        "coverage_all": sum(row["strict_48_m5_available"] for row in ledger) / len(ledger),
        "coverage_allowed_pairs": sum(row["strict_48_m5_available"] for row in eligible) / len(eligible),
        "reasons": dict(Counter(row["coverage_reason"] for row in ledger if row["coverage_reason"])),
        "parent_reason_readback": reasons,
        "ledger_path": str(path.relative_to(REPO)), "ledger_sha256": sha256(path),
    }
    write_json(ROOT / "gap_audit_v2.json", report)
    return report


def evaluate(build: dict[str, Any], gap_audit: dict[str, Any]) -> dict[str, Any]:
    parent = import_module("gapless_evaluation_parent", REPO / "research/historical_learning_price_action_admission/run_price_action_admission.py")
    parent.SOURCES = {
        pair: (source["s5_path"], source["s5_sha256"])
        for pair, source in build["sources"].items()
    }
    report = parent.run(REPO)
    report["contract"] = "historical_learning_gapless_truth_result_v2"
    report["preregister_sha256"] = sha256(PREREG)
    report["source_boundary"] = {
        "features": "DUKASCOPY_DATAFEED_TICK", "fills_and_labels": "OANDA_ACTUAL_AFTER_COST",
        "cross_source_same_truth_claimed": False,
    }
    report["episode_coverage_audit"] = gap_audit
    report["feature_coverage"]["reasons"] = {
        "AVAILABLE": gap_audit["strict_available_all"],
        "PAIR_OUT_OF_SCOPE_BY_EXPLICIT_ALLOWLIST": gap_audit["reasons"].get("PAIR_OUT_OF_SCOPE_BY_EXPLICIT_ALLOWLIST", 0),
        "MARKET_CLOSED_LOOKBACK_EXCLUDED": gap_audit["reasons"].get("MARKET_CLOSED_LOOKBACK_EXCLUDED", 0),
        "GAP_OR_LOOKBACK_INCOMPLETE": gap_audit["reasons"].get("GAP_OR_LOOKBACK_INCOMPLETE", 0),
    }
    scope_complete = gap_audit["episodes_allowed_pairs"] == gap_audit["episodes_all"]
    report["scope_decision"] = "PAIR_SCOPE_MATCHES_ALL_EPISODES" if scope_complete else "251/251_NOT_POSSIBLE_WITH_FROZEN_THREE_PAIR_ALLOWLIST"
    report["parent_model_decision"] = report.get("overall_decision")
    report["overall_decision"] = "ACCEPT" if report["parent_model_decision"] == "ACCEPT" and scope_complete else "REJECT"
    report["holdout_used"] = False
    output = ROOT / "report_v2.json"
    write_json(output, report)
    return report


def run_all(workers: int) -> dict[str, Any]:
    manifest = acquire(workers)
    gates = (
        "all_required_hours_complete", "market_open_defect_free",
        "bid_ask_valid", "double_download_match",
    )
    if not all(manifest["admission"][gate] for gate in gates):
        result = {"status": "BLOCKED_ACQUISITION_ADMISSION", "manifest": str((CACHE / "manifest_v2.json").relative_to(REPO)), "admission": manifest["admission"]}
        write_json(ROOT / "run_status_v2.json", result)
        return result
    build = build_s5()
    gap_audit = episode_lineage(build)
    report = evaluate(build, gap_audit)
    result = {
        "status": "COMPLETE", "overall_decision": report["overall_decision"], "holdout_used": False,
        "coverage_all": gap_audit["coverage_all"], "coverage_allowed_pairs": gap_audit["coverage_allowed_pairs"],
        "cache_bytes": directory_bytes(CACHE), "report": str((ROOT / "report_v2.json").relative_to(REPO)),
    }
    write_json(ROOT / "run_status_v2.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "acquire", "build", "evaluate", "all"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "plan":
        episodes = read_episodes()
        result = {"episodes": len(episodes), "allowed_pair_episodes": sum(row["pair"] in PAIRS for row in episodes), "required_hours": len(required_hours(episodes)), "cache_bytes": directory_bytes(CACHE)}
    elif args.command == "acquire":
        result = acquire(args.workers)
    elif args.command == "build":
        result = build_s5()
    elif args.command == "evaluate":
        build = json.loads((CACHE / "build_report_v2.json").read_text())
        result = evaluate(build, episode_lineage(build))
    else:
        result = run_all(args.workers)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("status") != "BLOCKED_ACQUISITION_ADMISSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
