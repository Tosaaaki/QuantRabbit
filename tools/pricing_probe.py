#!/usr/bin/env python3
"""Short-burst pricing probe for live FX microstructure.

Uses the OANDA pricing stream when possible, bounded to a few seconds so it
stays session-scoped rather than becoming a persistent daemon. Falls back to
pricing snapshots if the stream is unavailable.

The goal is to show whether the tape is:
- one-sided and clean
- two-way / whipsaw
- friction-dominated because spread is blowing out

Usage:
    python3 tools/pricing_probe.py
    python3 tools/pricing_probe.py --pairs EUR_USD,GBP_USD --samples 6 --interval 0.35
    python3 tools/pricing_probe.py --mode stream --pairs EUR_USD --seconds 2.0
    python3 tools/pricing_probe.py --pairs EUR_USD --json
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config_loader import get_oanda_config

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
PROBE_CACHE_PATH = ROOT / "logs" / "pricing_probe.json"
MIN_TAPE_SAMPLES = 3


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("JPY") else 10000


def _oanda_pricing(cfg: dict[str, object], pairs: list[str]) -> dict:
    acct = str(cfg["oanda_account_id"])
    params = urllib.parse.urlencode({"instruments": ",".join(pairs)})
    url = f"{cfg['oanda_base_url']}/v3/accounts/{acct}/pricing?{params}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {cfg['oanda_token']}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _oanda_stream_url(cfg: dict[str, object], pairs: list[str]) -> str:
    acct = str(cfg["oanda_account_id"])
    params = urllib.parse.urlencode({"instruments": ",".join(pairs)})
    base = str(cfg.get("oanda_stream_url") or "").rstrip("/")
    if not base:
        base = str(cfg["oanda_base_url"]).replace("api-fx", "stream-fx")
    return f"{base}/v3/accounts/{acct}/pricing/stream?{params}"


def _socket_set_timeout(resp, timeout_sec: float) -> None:
    for attr_chain in (
        ("fp", "raw", "_sock"),
        ("fp", "raw", "_fp", "fp", "raw", "_sock"),
    ):
        cur = resp
        try:
            for attr in attr_chain:
                cur = getattr(cur, attr)
            cur.settimeout(timeout_sec)
            return
        except Exception:
            continue


def _row_from_price_item(pair: str, item: dict, sample_time: str) -> dict | None:
    try:
        bid = float(item["bids"][0]["price"])
        ask = float(item["asks"][0]["price"])
    except Exception:
        return None
    return {
        "time": sample_time,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread_pips": (ask - bid) * _pip_factor(pair),
    }


def _probe_market_snapshot(
    cfg: dict[str, object],
    *,
    pairs: list[str],
    samples: int,
    interval_sec: float,
) -> tuple[dict[str, list[dict]], list[str], str]:
    snapshots: dict[str, list[dict]] = {pair: [] for pair in pairs}
    errors: list[str] = []
    next_deadline = time.monotonic()
    sample_count = max(1, samples)
    for idx in range(sample_count):
        try:
            data = _oanda_pricing(cfg, pairs)
            sample_time = datetime.now(timezone.utc).isoformat()
            seen = set()
            for item in data.get("prices", []):
                pair = str(item.get("instrument", ""))
                if pair not in snapshots:
                    continue
                row = _row_from_price_item(pair, item, sample_time)
                if row is None:
                    continue
                seen.add(pair)
                snapshots[pair].append(row)
            missing = [pair for pair in pairs if pair not in seen]
            if missing:
                errors.append(f"missing pricing rows for {', '.join(missing)}")
        except Exception as exc:
            errors.append(str(exc))

        if idx == sample_count - 1:
            continue
        next_deadline += interval_sec
        sleep_for = next_deadline - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    return snapshots, errors, "snapshot"


def _probe_market_stream(
    cfg: dict[str, object],
    *,
    pairs: list[str],
    duration_sec: float,
    read_timeout_sec: float = 0.40,
) -> tuple[dict[str, list[dict]], list[str], str]:
    snapshots: dict[str, list[dict]] = {pair: [] for pair in pairs}
    errors: list[str] = []
    url = _oanda_stream_url(cfg, pairs)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {cfg['oanda_token']}"},
    )
    deadline = time.monotonic() + max(0.5, duration_sec)
    try:
        with urllib.request.urlopen(req, timeout=max(5.0, duration_sec + 2.0)) as resp:
            _socket_set_timeout(resp, read_timeout_sec)
            while time.monotonic() < deadline:
                try:
                    raw = resp.readline()
                except socket.timeout:
                    continue
                except TimeoutError:
                    continue
                except OSError as exc:
                    if "timed out" in str(exc).lower():
                        continue
                    errors.append(str(exc))
                    break
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors.append("stream decode error")
                    continue
                if item.get("type") != "PRICE":
                    continue
                pair = str(item.get("instrument", ""))
                if pair not in snapshots:
                    continue
                sample_time = item.get("time") or datetime.now(timezone.utc).isoformat()
                row = _row_from_price_item(pair, item, sample_time)
                if row is None:
                    continue
                snapshots[pair].append(row)
    except Exception as exc:
        errors.append(str(exc))
    return snapshots, errors, "stream"


def _summarize_pair(pair: str, rows: list[dict]) -> dict:
    pip_factor = _pip_factor(pair)
    mids = [float(row["mid"]) for row in rows]
    spreads = [float(row["spread_pips"]) for row in rows]
    start_mid = mids[0]
    end_mid = mids[-1]
    delta_pips = (end_mid - start_mid) * pip_factor
    range_pips = (max(mids) - min(mids)) * pip_factor
    upticks = 0
    downticks = 0
    flats = 0
    for prev, cur in zip(mids, mids[1:]):
        if cur > prev:
            upticks += 1
        elif cur < prev:
            downticks += 1
        else:
            flats += 1

    avg_spread = sum(spreads) / len(spreads)
    max_spread = max(spreads)
    min_spread = min(spreads)
    spread_jump = max_spread - min_spread
    net_bias = upticks - downticks
    if delta_pips > 0.8 and upticks >= downticks + 2:
        bias = "buyers pressing"
    elif delta_pips < -0.8 and downticks >= upticks + 2:
        bias = "sellers pressing"
    elif upticks >= 2 and downticks >= 2:
        bias = "two-way"
    else:
        bias = "mixed"

    quiet_stable_spread_cap = 2.4 if pair.endswith("JPY") else 1.2
    quiet_stable_jump_cap = max(0.20, avg_spread * 0.30)

    if max_spread >= max(avg_spread * 2.2, 4.0 if pair.endswith("JPY") else 1.6):
        tape = "spread unstable"
    elif range_pips <= avg_spread * 1.2:
        if avg_spread <= quiet_stable_spread_cap and spread_jump <= quiet_stable_jump_cap:
            tape = "quiet / stable"
        else:
            tape = "friction-dominated"
    elif upticks >= 3 and downticks == 0:
        tape = "clean one-way"
    elif downticks >= 3 and upticks == 0:
        tape = "clean one-way"
    elif upticks >= 2 and downticks >= 2 and abs(delta_pips) <= avg_spread:
        tape = "whipsaw / two-way"
    else:
        tape = "tradeable"

    return {
        "pair": pair,
        "samples": len(rows),
        "start_mid": start_mid,
        "end_mid": end_mid,
        "delta_pips": round(delta_pips, 2),
        "range_pips": round(range_pips, 2),
        "avg_spread_pips": round(avg_spread, 2),
        "max_spread_pips": round(max_spread, 2),
        "spread_jump_pips": round(spread_jump, 2),
        "upticks": upticks,
        "downticks": downticks,
        "flats": flats,
        "net_bias": net_bias,
        "bias": bias,
        "tape": tape,
        "last_time": rows[-1]["time"],
    }


def probe_market(
    cfg: dict[str, object] | None = None,
    *,
    pairs: list[str] | None = None,
    mode: str = "stream",
    samples: int = 6,
    interval_sec: float = 0.35,
    duration_sec: float | None = None,
    write_cache: bool = True,
) -> dict:
    cfg = cfg or get_oanda_config()
    target_pairs = pairs or list(PAIRS)
    started_at = datetime.now(timezone.utc)
    duration = duration_sec if duration_sec is not None else max(interval_sec * max(samples - 1, 1), 1.2)
    probe_mode = mode.lower().strip()
    mode_used = probe_mode
    if probe_mode == "stream":
        snapshots, errors, mode_used = _probe_market_stream(
            cfg,
            pairs=target_pairs,
            duration_sec=duration,
        )
        missing_pairs = [pair for pair, rows in snapshots.items() if len(rows) < MIN_TAPE_SAMPLES]
        if missing_pairs:
            snap_rows, snap_errors, snap_mode = _probe_market_snapshot(
                cfg,
                pairs=missing_pairs,
                samples=samples,
                interval_sec=interval_sec,
            )
            for pair, rows in snap_rows.items():
                if len(snapshots.get(pair, [])) < MIN_TAPE_SAMPLES:
                    snapshots[pair] = rows
            if snap_errors:
                errors.extend([f"stream fallback: {err}" for err in snap_errors])
            if missing_pairs:
                errors.append(f"stream fallback used for {', '.join(missing_pairs)}")
            if len(missing_pairs) == len(target_pairs):
                mode_used = f"{mode_used}->{snap_mode}"
            else:
                mode_used = f"{mode_used}+{snap_mode}"
    else:
        snapshots, errors, mode_used = _probe_market_snapshot(
            cfg,
            pairs=target_pairs,
            samples=samples,
            interval_sec=interval_sec,
        )

    fetched_at = datetime.now(timezone.utc)
    summaries = {}
    for pair, rows in snapshots.items():
        if len(rows) >= 2:
            summaries[pair] = _summarize_pair(pair, rows)
        else:
            summaries[pair] = {
                "pair": pair,
                "samples": len(rows),
                "bias": "unknown",
                "tape": "unavailable",
                "error": "insufficient samples",
            }

    payload = {
        "started_at": started_at.isoformat(),
        "fetched_at": fetched_at.isoformat(),
        "duration_sec": round((fetched_at - started_at).total_seconds(), 2),
        "mode_requested": probe_mode,
        "mode_used": mode_used,
        "samples_requested": samples,
        "interval_sec": interval_sec,
        "stream_duration_sec": round(duration, 2),
        "pairs": summaries,
        "errors": errors,
    }
    if write_cache:
        PROBE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROBE_CACHE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def execution_guard(summary: dict, *, order_type: str) -> str | None:
    if order_type != "MARKET":
        return None
    samples = int(summary.get("samples", 0) or 0)
    if samples < MIN_TAPE_SAMPLES:
        return f"{summary.get('pair')} live tape probe has only {samples} samples"
    tape = str(summary.get("tape") or "")
    if tape == "spread unstable":
        return (
            f"{summary.get('pair')} live tape probe shows spread instability "
            f"(avg {summary.get('avg_spread_pips')}pip / max {summary.get('max_spread_pips')}pip)"
        )
    if tape == "friction-dominated":
        return (
            f"{summary.get('pair')} live tape probe is friction-dominated "
            f"(range {summary.get('range_pips')}pip vs avg spread {summary.get('avg_spread_pips')}pip)"
        )
    return None


def _format_summary_line(summary: dict) -> str:
    if summary.get("tape") == "unavailable":
        return f"{summary['pair']}: unavailable ({summary.get('error', 'no data')})"
    return (
        f"{summary['pair']}: {summary['bias']} | tape={summary['tape']} | "
        f"move={summary['delta_pips']:+.1f}pip | range={summary['range_pips']:.1f}pip | "
        f"spread avg/max={summary['avg_spread_pips']:.1f}/{summary['max_spread_pips']:.1f}pip | "
        f"ticks {summary['upticks']}/{summary['downticks']}/{summary['flats']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--mode", choices=["stream", "snapshot"], default="stream")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pairs = [token.strip().upper() for token in args.pairs.split(",") if token.strip()]
    payload = probe_market(
        pairs=pairs,
        mode=args.mode,
        samples=args.samples,
        interval_sec=args.interval,
        duration_sec=args.seconds,
        write_cache=True,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"Pricing probe {payload['duration_sec']:.2f}s "
        f"(mode={payload['mode_used']} | {payload['samples_requested']} samples x {payload['interval_sec']:.2f}s)"
    )
    for pair in pairs:
        print(_format_summary_line(payload["pairs"][pair]))
    if payload["errors"]:
        print("errors:")
        for error in payload["errors"][:5]:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
