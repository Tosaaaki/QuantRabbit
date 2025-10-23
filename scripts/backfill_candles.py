#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_candles.py
~~~~~~~~~~~~~~~~~~~

リプレイログ (logs/replay/...) もしくは OANDA API から日次のローソク JSON
(`logs/candles_<TF>_<YYYYMMDD>.json`) を再生成する補助スクリプト。

Usage:
    python scripts/backfill_candles.py --start-date 2025-10-01 --end-date 2025-10-22 --timeframes M1,H4 --fetch-missing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.secrets import get_secret
DEFAULT_REPLAY_DIR = REPO_ROOT / "logs" / "replay"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs"
SUPPORTED_TIMEFRAMES = {"M1", "H4"}


@dataclass(frozen=True)
class Config:
    instrument: str
    timeframes: Sequence[str]
    start: Optional[date]
    end: Optional[date]
    replay_dir: Path
    output_dir: Path
    overwrite: bool
    fetch_missing: bool


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> List[date]:
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    delta = (end - start).days
    return [start + timedelta(days=i) for i in range(delta + 1)]


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_time_for_oanda(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


def _format_price(value: float) -> str:
    return f"{value:.3f}"


def _load_replay_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda row: row.get("ts", ""))
    return rows


def _build_payload_from_replay(
    instrument: str, timeframe: str, rows: List[Dict[str, object]]
) -> Dict[str, object]:
    candles: List[Dict[str, object]] = []
    for row in rows:
        ts_raw = row.get("ts")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except ValueError:
            continue

        o = row.get("open")
        h = row.get("high")
        l = row.get("low")
        c = row.get("close")
        if None in (o, h, l, c):
            continue

        volume = row.get("volume")
        try:
            volume_int = int(volume) if volume is not None else 0
        except (TypeError, ValueError):
            volume_int = 0

        candles.append(
            {
                "complete": True,
                "volume": volume_int,
                "time": _format_time_for_oanda(ts),
                "mid": {
                    "o": _format_price(float(o)),
                    "h": _format_price(float(h)),
                    "l": _format_price(float(l)),
                    "c": _format_price(float(c)),
                },
            }
        )

    payload = {
        "instrument": instrument,
        "granularity": timeframe,
        "candles": candles,
    }
    return payload


def _oanda_host() -> str:
    token = get_secret("oanda_token")
    try:
        pract = str(get_secret("oanda_practice")).lower() == "true"
    except Exception:
        pract = False
    base = "https://api-fxpractice.oanda.com" if pract else "https://api-fxtrade.oanda.com"
    return base, token


def _fetch_oanda_candles(
    instrument: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[List[Dict[str, object]]]:
    base_url, token = _oanda_host()
    url = f"{base_url}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": timeframe,
        "price": "M",
        "from": start_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "to": end_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("candles", [])
    except Exception as exc:
        print(f"[WARN] failed to fetch OANDA candles ({instrument} {timeframe} {start_dt:%Y-%m-%d}): {exc}")
        return None


def _build_payload_from_oanda(
    instrument: str,
    timeframe: str,
    candles: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    converted: List[Dict[str, object]] = []
    for candle in candles:
        t = candle.get("time")
        if not t:
            continue
        converted.append(
            {
                "complete": bool(candle.get("complete", True)),
                "volume": candle.get("volume", 0),
                "time": str(t),
                "mid": candle.get("mid") or candle.get("midpoint") or {},
            }
        )
    return {
        "instrument": instrument,
        "granularity": timeframe,
        "candles": converted,
    }


def _resolve_dates(cfg: Config) -> List[date]:
    if cfg.start and cfg.end:
        return _date_range(cfg.start, cfg.end)
    replay_dates: set[date] = set()
    tf_dir = cfg.replay_dir / cfg.instrument
    if not tf_dir.exists():
        return []
    for tf in cfg.timeframes:
        prefix = f"{cfg.instrument}_{tf}_"
        for path in tf_dir.glob(f"{prefix}*.jsonl"):
            stem = path.stem
            if not stem.startswith(prefix):
                continue
            day_token = stem[len(prefix) :]
            if len(day_token) != 8 or not day_token.isdigit():
                continue
            replay_dates.add(_parse_date(f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:]}"))
    if cfg.start and replay_dates:
        filtered = [d for d in replay_dates if d >= cfg.start]
        return sorted(filtered)
    return sorted(replay_dates)


def process(cfg: Config) -> None:
    _ensure_output_dir(cfg.output_dir)
    dates = _resolve_dates(cfg)
    if cfg.start and cfg.end and not dates:
        dates = _date_range(cfg.start, cfg.end)
    if not dates:
        print("[WARN] No target dates found. Nothing to do.")
        return

    now_utc = datetime.utcnow().date()

    for day in dates:
        print(f"[INFO] processing {day.isoformat()}...")
        for tf in cfg.timeframes:
            if tf not in SUPPORTED_TIMEFRAMES:
                print(f"[WARN] unsupported timeframe: {tf}. skip.")
                continue
            replay_path = cfg.replay_dir / cfg.instrument / f"{cfg.instrument}_{tf}_{day.strftime('%Y%m%d')}.jsonl"
            out_path = cfg.output_dir / f"candles_{tf}_{day.strftime('%Y%m%d')}.json"

            if out_path.exists() and not cfg.overwrite:
                print(f"  [SKIP] {out_path.name} already exists.")
                continue

            if replay_path.exists():
                rows = _load_replay_jsonl(replay_path)
                payload = _build_payload_from_replay(cfg.instrument, tf, rows)
                with out_path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                print(f"  [OK] wrote {out_path.name} from replay ({len(payload['candles'])} candles).")
                continue

            if not cfg.fetch_missing:
                print(f"  [MISS] replay not found for {tf} {day}. use --fetch-missing to download.")
                continue

            if day >= now_utc:
                print(f"  [MISS] skip future date {day}.")
                continue

            start_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            if end_dt.date() >= now_utc:
                end_dt = datetime.utcnow().replace(tzinfo=timezone.utc)

            candles = _fetch_oanda_candles(cfg.instrument, tf, start_dt, end_dt)
            if not candles:
                print(f"  [MISS] no candles retrieved for {tf} {day}.")
                continue

            payload = _build_payload_from_oanda(cfg.instrument, tf, candles)
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            print(f"  [OK] wrote {out_path.name} from OANDA ({len(payload['candles'])} candles).")


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Replay/OANDA からローソク JSON を再生成します")
    ap.add_argument("--instrument", default="USD_JPY", help="対象銘柄 (既定: USD_JPY)")
    ap.add_argument("--timeframes", default="M1", help="カンマ区切りで TF 指定 (例: M1,H4)")
    ap.add_argument("--start-date", help="YYYY-MM-DD 形式。省略時はリプレイから推定")
    ap.add_argument("--end-date", help="YYYY-MM-DD 形式。省略時は start-date と同じ扱い")
    ap.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR), help="リプレイログのフォルダ")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="出力先ディレクトリ (logs 配下推奨)")
    ap.add_argument("--overwrite", action="store_true", help="既存ファイルを上書き")
    ap.add_argument("--fetch-missing", action="store_true", help="リプレイが無い場合は OANDA API から補完")
    args = ap.parse_args()

    start = _parse_date(args.start_date) if args.start_date else None
    if args.end_date:
        end = _parse_date(args.end_date)
    elif start:
        end = start
    else:
        end = None

    cfg = Config(
        instrument=args.instrument,
        timeframes=[tf.strip() for tf in args.timeframes.split(",") if tf.strip()],
        start=start,
        end=end,
        replay_dir=Path(args.replay_dir),
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
        fetch_missing=args.fetch_missing,
    )
    return cfg


def main() -> None:
    cfg = parse_args()
    process(cfg)


if __name__ == "__main__":
    main()
