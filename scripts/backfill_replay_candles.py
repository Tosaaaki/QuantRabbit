#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill replay candle logs from OANDA REST API.

This script downloads historical candles (M5/H1/D1 etc.) and appends them into
QuantRabbit replay logs under:
  logs/replay/<instrument>/<instrument>_<TF>_YYYYMMDD.jsonl

These replay logs are used by scripts/train_forecast_bundle.py (and other
offline analyzers) as an "offline-friendly" historical candle source.

Notes
-----
- This is best-effort and idempotent: it skips candles whose ts already exists
  in the target day file.
- It does NOT touch secrets; it reads OANDA credentials via utils.secrets.
- It only writes under logs/replay/ by default.

Examples (on VM):
  python scripts/backfill_replay_candles.py --instrument USD_JPY --timeframes M5,H1,D1
  python scripts/backfill_replay_candles.py --instrument USD_JPY --timeframes M5,H1,D1 --lookback-h1-days 900
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.secrets import get_secret


SUPPORTED_TIMEFRAMES = ("M5", "H1", "D1")
_OANDA_GRANULARITY_MAP = {
    "D1": "D",
}


def _parse_time(value: str) -> dt.datetime:
    """Convert OANDA timestamp (nanosecond precision) into datetime."""
    iso = value.replace("Z", "+00:00")
    if "." not in iso:
        ts = dt.datetime.fromisoformat(iso)
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)

    head, frac_and_tz = iso.split(".", 1)
    tz = "+00:00"
    if "+" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("+", 1)
        tz = "+" + tz_tail
    elif "-" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("-", 1)
        tz = "-" + tz_tail
    else:
        frac = frac_and_tz

    frac = (frac[:6]).ljust(6, "0")
    ts = dt.datetime.fromisoformat(f"{head}.{frac}{tz}")
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)


def _iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_dt(text: str) -> dt.datetime:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty_datetime")
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        # YYYY-MM-DD
        return dt.datetime.fromisoformat(raw).replace(tzinfo=dt.timezone.utc)
    raw = raw.replace("Z", "+00:00")
    ts = dt.datetime.fromisoformat(raw)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _oanda_host_and_token() -> Tuple[str, str]:
    token = get_secret("oanda_token")
    try:
        pract = str(get_secret("oanda_practice")).lower() == "true"
    except Exception:
        pract = False
    host = "https://api-fxpractice.oanda.com" if pract else "https://api-fxtrade.oanda.com"
    return host, token


@dataclass(frozen=True)
class BackfillTarget:
    timeframe: str
    start: dt.datetime
    end: dt.datetime


def _discover_targets(
    timeframes: Iterable[str],
    *,
    start: Optional[dt.datetime],
    end: dt.datetime,
    lookback_m5_days: int,
    lookback_h1_days: int,
    lookback_d1_days: int,
) -> List[BackfillTarget]:
    out: List[BackfillTarget] = []
    for tf in timeframes:
        tf_norm = tf.strip().upper()
        if tf_norm not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"unsupported_timeframe:{tf_norm} (supported={','.join(SUPPORTED_TIMEFRAMES)})")
        if start is not None:
            tf_start = start
        else:
            if tf_norm == "M5":
                tf_start = end - dt.timedelta(days=max(1, int(lookback_m5_days)))
            elif tf_norm == "H1":
                tf_start = end - dt.timedelta(days=max(1, int(lookback_h1_days)))
            else:  # D1
                tf_start = end - dt.timedelta(days=max(1, int(lookback_d1_days)))
        out.append(BackfillTarget(timeframe=tf_norm, start=tf_start, end=end))
    return out


def _load_seen_ts(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ts = obj.get("ts") or obj.get("timestamp") or obj.get("time")
                if ts:
                    seen.add(str(ts))
    except FileNotFoundError:
        return seen
    except Exception:
        # best-effort
        return seen
    return seen


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            n += 1
    return n


def _fetch_oanda_page(
    *,
    client: httpx.Client,
    host: str,
    token: str,
    instrument: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    include_first: bool,
    timeout_sec: float,
    max_retries: int,
) -> List[dict]:
    gran = _OANDA_GRANULARITY_MAP.get(timeframe, timeframe)
    url = f"{host}/v3/instruments/{instrument}/candles"
    params = {
        "price": "M",
        "granularity": gran,
        "from": _iso_z(start),
        "to": _iso_z(end),
        "includeFirst": "true" if include_first else "false",
    }
    headers = {"Authorization": f"Bearer {token}"}

    last_exc: Exception | None = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            resp = client.get(url, headers=headers, params=params, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("candles") or []
            return list(candles) if isinstance(candles, list) else []
        except Exception as exc:
            last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
            if attempt >= max_retries:
                break
            time.sleep(min(2.0, 0.25 * attempt))
    raise RuntimeError(f"oanda_fetch_failed:{timeframe}:{last_exc}")


def _chunk_days_for(timeframe: str, *, chunk_m5_days: int, chunk_h1_days: int, chunk_d1_days: int) -> int:
    tf = (timeframe or "").strip().upper()
    if tf == "M5":
        return max(1, int(chunk_m5_days))
    if tf == "H1":
        return max(1, int(chunk_h1_days))
    return max(1, int(chunk_d1_days))


def _iter_oanda_candles(
    *,
    client: httpx.Client,
    host: str,
    token: str,
    instrument: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    chunk_m5_days: int,
    chunk_h1_days: int,
    chunk_d1_days: int,
    sleep_sec: float,
    timeout_sec: float,
    max_retries: int,
) -> Iterator[Tuple[dt.datetime, dict]]:
    """
    Yield (ts, normalized_row) over [start, end) by fetching in time windows.

    OANDA candles endpoint is reliable with from/to windows. Chunk sizes are chosen
    to avoid hitting API limits (e.g., 5000 candles per response).
    """
    cursor = start
    include_first = True
    chunk_days = _chunk_days_for(
        timeframe,
        chunk_m5_days=chunk_m5_days,
        chunk_h1_days=chunk_h1_days,
        chunk_d1_days=chunk_d1_days,
    )
    chunk = dt.timedelta(days=chunk_days)

    while cursor < end:
        window_end = min(cursor + chunk, end)
        raw = _fetch_oanda_page(
            client=client,
            host=host,
            token=token,
            instrument=instrument,
            timeframe=timeframe,
            start=cursor,
            end=window_end,
            include_first=include_first,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
        )
        include_first = False
        for c in raw:
            try:
                if c.get("complete") is False:
                    continue
                ts = _parse_time(str(c["time"]))
                mid = c.get("mid") or {}
                row = {
                    "ts": _iso_z(ts),
                    "timeframe": timeframe,
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume") or 0),
                }
            except Exception:
                continue
            yield ts, row
        cursor = window_end
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))


def backfill_replay_logs(
    *,
    instrument: str,
    targets: Sequence[BackfillTarget],
    replay_dir: Path,
    chunk_m5_days: int,
    chunk_h1_days: int,
    chunk_d1_days: int,
    sleep_sec: float,
    timeout_sec: float,
    max_retries: int,
    dry_run: bool,
) -> None:
    host, token = _oanda_host_and_token()
    base_dir = replay_dir / instrument
    base_dir.mkdir(parents=True, exist_ok=True)

    seen_cache: Dict[Path, set[str]] = {}

    with httpx.Client() as client:
        for t in targets:
            tf = t.timeframe
            print(
                f"[BACKFILL] tf={tf} instrument={instrument} start={_iso_z(t.start)} end={_iso_z(t.end)}"
            )
            fetched = 0
            written = 0
            skipped = 0

            pending_by_path: Dict[Path, List[dict]] = {}
            for ts, row in _iter_oanda_candles(
                client=client,
                host=host,
                token=token,
                instrument=instrument,
                timeframe=tf,
                start=t.start,
                end=t.end,
                chunk_m5_days=chunk_m5_days,
                chunk_h1_days=chunk_h1_days,
                chunk_d1_days=chunk_d1_days,
                sleep_sec=sleep_sec,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
            ):
                fetched += 1
                day = ts.astimezone(dt.timezone.utc).strftime("%Y%m%d")
                path = base_dir / f"{instrument}_{tf}_{day}.jsonl"
                if path not in seen_cache:
                    seen_cache[path] = _load_seen_ts(path)
                key = row.get("ts")
                if key and key in seen_cache[path]:
                    skipped += 1
                    continue
                if key:
                    seen_cache[path].add(str(key))
                pending_by_path.setdefault(path, []).append(row)

                # Flush in chunks to keep memory bounded.
                if sum(len(v) for v in pending_by_path.values()) >= 2000:
                    for fp, rows in list(pending_by_path.items()):
                        if not rows:
                            continue
                        if dry_run:
                            written += len(rows)
                        else:
                            written += _append_jsonl(fp, rows)
                        pending_by_path[fp] = []

            # Final flush
            for fp, rows in pending_by_path.items():
                if not rows:
                    continue
                if dry_run:
                    written += len(rows)
                else:
                    written += _append_jsonl(fp, rows)

            print(
                f"[BACKFILL] tf={tf} fetched={fetched} written={written} skipped={skipped} dry_run={int(dry_run)}"
            )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="USD_JPY")
    ap.add_argument(
        "--timeframes",
        default="M5,H1,D1",
        help=f"comma-separated list (default: M5,H1,D1; supported: {','.join(SUPPORTED_TIMEFRAMES)})",
    )
    ap.add_argument("--replay-dir", type=Path, default=REPO_ROOT / "logs" / "replay")
    ap.add_argument("--start", default="", help="ISO datetime or YYYY-MM-DD (UTC). Overrides lookback-days.")
    ap.add_argument("--end", default="", help="ISO datetime or YYYY-MM-DD (UTC). Default: now (UTC)")
    ap.add_argument("--lookback-m5-days", type=int, default=120)
    ap.add_argument("--lookback-h1-days", type=int, default=540)
    ap.add_argument("--lookback-d1-days", type=int, default=2000)

    ap.add_argument(
        "--chunk-m5-days",
        type=int,
        default=14,
        help="Fetch window size for M5 backfill (default: 14 days; tuned to stay under API limits).",
    )
    ap.add_argument(
        "--chunk-h1-days",
        type=int,
        default=180,
        help="Fetch window size for H1 backfill (default: 180 days; tuned to stay under API limits).",
    )
    ap.add_argument(
        "--chunk-d1-days",
        type=int,
        default=2000,
        help="Fetch window size for D1 backfill (default: 2000 days).",
    )
    ap.add_argument("--sleep-sec", type=float, default=0.1, help="Sleep between windows (default: 0.1s)")
    ap.add_argument("--timeout-sec", type=float, default=12.0, help="HTTP timeout seconds (default: 12)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="Fetch and count only; do not write files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tfs = [t.strip() for t in str(args.timeframes).split(",") if t.strip()]
    end = _parse_dt(args.end) if str(args.end).strip() else dt.datetime.now(dt.timezone.utc)
    start = _parse_dt(args.start) if str(args.start).strip() else None

    targets = _discover_targets(
        tfs,
        start=start,
        end=end,
        lookback_m5_days=int(args.lookback_m5_days),
        lookback_h1_days=int(args.lookback_h1_days),
        lookback_d1_days=int(args.lookback_d1_days),
    )
    backfill_replay_logs(
        instrument=str(args.instrument).strip().upper(),
        targets=targets,
        replay_dir=Path(args.replay_dir),
        chunk_m5_days=max(1, int(args.chunk_m5_days)),
        chunk_h1_days=max(1, int(args.chunk_h1_days)),
        chunk_d1_days=max(1, int(args.chunk_d1_days)),
        sleep_sec=max(0.0, float(args.sleep_sec)),
        timeout_sec=max(1.0, float(args.timeout_sec)),
        max_retries=max(1, int(args.max_retries)),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
