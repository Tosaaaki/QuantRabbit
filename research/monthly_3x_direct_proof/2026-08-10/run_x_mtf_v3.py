#!/usr/bin/env python3
"""Strict one-variable execution of Q-XFX-MTF-001.

Research-only: reads one archived OANDA S5 bid/ask gzip and never imports an
order/broker client.  The baseline entry and financial replay are reused from
the frozen X LVN experiment.  The sole candidate change is rejection of a
lower-timeframe failed-break signal when three *fully observed* completed H1
bars show a structural continuation in the opposite direction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LVN_PATH = REPO / "research/x_fx_methods/2026-08-09/run_lvn_filter_experiment.py"
SOURCE_DEFAULT = REPO / "logs/replay/oanda_history/20260705T081428Z/USD_JPY/USD_JPY_S5_BA_20260608T060528Z_20260626T065906Z.jsonl.gz"
EXPECTED_SOURCE_SHA256 = "c46612964813f1b5fdd8235b8703e0d70e1be784412479edbc7126c5484d8f81"


def _load_lvn():
    spec = importlib.util.spec_from_file_location("x_lvn_frozen", LVN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen LVN module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
        if "." in text:
            head, frac = text.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text += "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def strict_h1_bars(path: Path, stop_before: datetime) -> tuple[list[dict], dict]:
    """Build H1 only from 12 M5 buckets each containing the exact 60 S5 slots."""
    m5_rows: dict[datetime, list[dict]] = defaultdict(list)
    raw_rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            timestamp = parse_utc(str(row["time"]))
            if timestamp >= stop_before:
                break
            raw_rows += 1
            bucket = timestamp.replace(
                minute=timestamp.minute - timestamp.minute % 5,
                second=0,
                microsecond=0,
            )
            row["_time"] = timestamp
            m5_rows[bucket].append(row)

    complete_m5: dict[datetime, dict] = {}
    for bucket, rows in m5_rows.items():
        expected = [bucket + timedelta(seconds=5 * index) for index in range(60)]
        actual = [row["_time"] for row in rows]
        if actual != expected:
            continue
        bids = [row["bid"] for row in rows]
        asks = [row["ask"] for row in rows]
        complete_m5[bucket] = {
            "time": bucket,
            "o": (float(bids[0]["o"]) + float(asks[0]["o"])) / 2,
            "h": max((float(b["h"]) + float(a["h"])) / 2 for b, a in zip(bids, asks)),
            "l": min((float(b["l"]) + float(a["l"])) / 2 for b, a in zip(bids, asks)),
            "c": (float(bids[-1]["c"]) + float(asks[-1]["c"])) / 2,
        }

    h1: list[dict] = []
    by_hour: dict[datetime, list[dict]] = defaultdict(list)
    for bar in complete_m5.values():
        by_hour[floor_hour(bar["time"])].append(bar)
    for hour, bars in sorted(by_hour.items()):
        bars.sort(key=lambda item: item["time"])
        expected = [hour + timedelta(minutes=5 * index) for index in range(12)]
        if [bar["time"] for bar in bars] != expected:
            continue
        h1.append({
            "start": hour,
            "end": hour + timedelta(hours=1),
            "o": bars[0]["o"],
            "h": max(bar["h"] for bar in bars),
            "l": min(bar["l"] for bar in bars),
            "c": bars[-1]["c"],
        })
    return h1, {
        "raw_s5_rows": raw_rows,
        "observed_m5_buckets": len(m5_rows),
        "strict_60_of_60_m5": len(complete_m5),
        "strict_12_of_12_h1": len(h1),
        "acquisition_gap_m5": len(m5_rows) - len(complete_m5),
    }


def parent_opinion(h1: list[dict], signal_time: datetime) -> str | None:
    completed = [bar for bar in h1 if bar["end"] <= signal_time]
    if len(completed) < 3:
        return None
    a, b, c = completed[-3:]
    if not (a["end"] == b["start"] and b["end"] == c["start"]):
        return None
    if c["c"] > max(a["h"], b["h"]) and a["c"] < b["c"] < c["c"] and c["c"] > c["o"]:
        return "LONG"
    if c["c"] < min(a["l"], b["l"]) and a["c"] > b["c"] > c["c"] and c["c"] < c["o"]:
        return "SHORT"
    return "NEUTRAL"


def run(source: Path) -> dict:
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("source hash differs from preregistered archive")
    lvn = _load_lvn()
    train_end = parse_utc("2026-06-17T00:00:00Z")
    validation_start = parse_utc("2026-06-18T00:00:00Z")
    validation_end = parse_utc("2026-06-23T00:00:00Z")

    m5, _, _ = lvn.aggregate_mid_bars(
        lvn.iter_s5(source, stop_before=validation_end), lvn.BAR_SECONDS
    )
    m30, _, _ = lvn.aggregate_mid_bars(
        lvn.iter_s5(source, stop_before=validation_end), lvn.PROFILE_SECONDS
    )
    signals = lvn.detect_signals(m5, lvn.build_prior_profiles(m30))
    outcomes, skipped = lvn.simulate(source, signals, stop_before=validation_end)
    train, validation = lvn.split_outcomes(
        outcomes, train_end=train_end, validation_start=validation_start
    )
    h1, coverage = strict_h1_bars(source, validation_end)

    def eligible(rows):
        result = []
        opinions = {}
        for row in rows:
            opinion = parent_opinion(h1, parse_utc(row.signal_time_utc))
            if opinion is not None:
                result.append(row)
                opinions[row.signal_time_utc] = opinion
        return result, opinions

    train_base, train_op = eligible(train)
    val_base, val_op = eligible(validation)
    train_candidate = [row for row in train_base if train_op[row.signal_time_utc] in {"NEUTRAL", row.side}]
    val_candidate = [row for row in val_base if val_op[row.signal_time_utc] in {"NEUTRAL", row.side}]
    paired = lvn.paired_daily_bootstrap(val_base, val_candidate)
    val_metrics = lvn.metrics(val_candidate)
    train_metrics = lvn.metrics(train_candidate)
    lcb = paired.get("lower_95pct")
    accepted = (
        train_metrics["trade_count"] >= 20
        and val_metrics["trade_count"] >= 10
        and (train_metrics["expectancy_jpy_per_1000u"] or 0) > 0
        and (val_metrics["expectancy_jpy_per_1000u"] or 0) > 0
        and isinstance(lcb, (int, float)) and lcb > 0
    )
    changed_train = len(train_base) - len(train_candidate)
    changed_val = len(val_base) - len(val_candidate)
    return {
        "schema_version": 1,
        "contract_id": "Q-XFX-MTF-001",
        "status": "RESEARCH_ACCEPTED" if accepted else "REJECT_OR_INSUFFICIENT_EVIDENCE",
        "monthly_3x_proven": False,
        "reason_monthly_3x": "The fixed archive does not contain a 30-day validation interval and cannot prove 200,000 to 600,000 JPY.",
        "source": {"path": str(source.relative_to(REPO)), "sha256": source_hash},
        "split": {
            "train_end": "2026-06-17T00:00:00Z",
            "validation_start": "2026-06-18T00:00:00Z",
            "validation_end": "2026-06-23T00:00:00Z",
            "embargo_hours": 24,
            "holdout_unread": True,
        },
        "classifier": {
            "mapping": "M5 entry -> H1 parent",
            "evidence": "three contiguous H1 bars, each built from 12 exact 60/60 M5 bars",
            "long": "latest close breaks prior two highs, three closes rising, bullish body",
            "short": "latest close breaks prior two lows, three closes falling, bearish body",
            "candidate_change": "reject only directionally opposite STRONG_CONTINUATION",
            "missing": "excluded from both paired arms, never treated as neutral",
            "function_sha256": hashlib.sha256(parent_opinion.__code__.co_code).hexdigest(),
        },
        "coverage": coverage,
        "counts": {
            "signals": len(signals), "outcomes": len(outcomes), "skipped": skipped,
            "train_eligible": len(train_base), "validation_eligible": len(val_base),
            "train_changed": changed_train, "validation_changed": changed_val,
        },
        "train": {"baseline": lvn.metrics(train_base), "candidate": train_metrics},
        "validation": {
            "baseline": lvn.metrics(val_base), "candidate": val_metrics,
            "paired_daily": paired,
        },
        "acceptance_gate": "TRAIN>=20, VALIDATION>=10, both expectancy>0, paired validation daily LCB>0",
        "limitations": [
            "Acquisition gaps are excluded rather than imputed.",
            "The frozen X handoff archive is shorter than 30 days.",
            "Fee and financing remain unsuitable for holds crossing a financing boundary; the inherited replay forces session end.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SOURCE_DEFAULT))
    parser.add_argument("--output", default=str(HERE / "x_mtf_result_v3.json"))
    args = parser.parse_args()
    payload = run(Path(args.input))
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "counts": payload["counts"], "validation": payload["validation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
