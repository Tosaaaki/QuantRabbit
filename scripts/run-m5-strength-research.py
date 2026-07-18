#!/usr/bin/env python3
"""M5-era opening research: 8-currency strength, 12 fixed candidates.

Signal: hourly mid closes from the sealed M5 corpus decompose into eight
currency strength returns (mean signed log-return across the pairs each
currency appears in).  Each candidate trades the strongest-vs-weakest
currency pair (or the top-2 basket) in the momentum direction.

Execution: real M5 bid/ask opens (spread paid), one non-overlapping
position set at a time, pre-entry continuous-hold weekend gate.  Money
accounting per the operator's standard: NAV 200,000 JPY, declared 10x
single-position leverage, per-trade cost stress, compounded daily.

Candidates (12, fixed): lookback {8h, 24h, 120h} x hold {4h, 8h} x
{TOP1, TOP2_BASKET}.  Splits: TRAIN [2020-01-01, 2024-01-01), VALIDATION
[2024-01-01, 2026-01-01), TEST [2026-01-01, 2026-07-10) UNTOUCHED here.
TRAIN runs a week-block-bootstrap White Reality Check across all twelve;
the single best stressed-positive TRAIN candidate replicates unchanged on
VALIDATION.  Shadow only; no order authority.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import random
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS

UTC = timezone.utc
TRAIN_FROM = datetime(2020, 1, 1, tzinfo=UTC)
TRAIN_TO = datetime(2024, 1, 1, tzinfo=UTC)
VAL_TO = datetime(2026, 1, 1, tzinfo=UTC)
TEST_TO = datetime(2026, 7, 10, tzinfo=UTC)
HOUR = 3600
LOOKBACKS_H = (8, 24, 120)
HOLDS_H = (4, 8)
VARIANTS = ("TOP1", "TOP2_BASKET")
MIN_PAIRS_PER_CURRENCY = 4
WEEKEND_MARGIN_MIN = 5
NAV_JPY = 200_000.0
LEVERAGE = 10.0
COST_STRESS_RETURN = 0.00005 * LEVERAGE  # ~0.5 pip on a USD pair, per trade
BOOT_BLOCK_DAYS = 5
BOOT_SAMPLES = 300
BOOT_SEED = 20260718
CURRENCIES = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_pair(root: Path, pair: str):
    epochs = array("q")
    bid_o = array("d")
    ask_o = array("d")
    hourly: dict[int, float] = {}
    for shard_file in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                stamp = row["time"]
                epoch = int(
                    datetime.fromisoformat(stamp[:19] + "+00:00").timestamp()
                )
                epochs.append(epoch)
                bid_o.append(float(row["bid"]["o"]))
                ask_o.append(float(row["ask"]["o"]))
                close_epoch = epoch + 300
                if close_epoch % HOUR == 0:
                    mid_close = (float(row["bid"]["c"]) + float(row["ask"]["c"])) / 2.0
                    hourly[close_epoch] = mid_close
    return epochs, bid_o, ask_o, hourly


def _strength_at(
    hour_epoch: int, lookback_h: int, hourly: dict[str, dict[int, float]]
) -> dict[str, float] | None:
    past = hour_epoch - lookback_h * HOUR
    sums: dict[str, float] = {c: 0.0 for c in CURRENCIES}
    counts: dict[str, int] = {c: 0 for c in CURRENCIES}
    for pair, closes in hourly.items():
        now = closes.get(hour_epoch)
        then = closes.get(past)
        if now is None or then is None or then <= 0:
            continue
        ret = math.log(now / then)
        base, quote = pair.split("_")
        sums[base] += ret
        counts[base] += 1
        sums[quote] -= ret
        counts[quote] += 1
    if any(counts[c] < MIN_PAIRS_PER_CURRENCY for c in CURRENCIES):
        return None
    return {c: sums[c] / counts[c] for c in CURRENCIES}


def _pair_and_side(strong: str, weak: str) -> tuple[str, str] | None:
    if f"{strong}_{weak}" in DEFAULT_TRADER_PAIRS:
        return f"{strong}_{weak}", "LONG"
    if f"{weak}_{strong}" in DEFAULT_TRADER_PAIRS:
        return f"{weak}_{strong}", "SHORT"
    return None


GROSS_MODE = False


def _execute(
    pair_data, pair: str, side: str, entry_after: int, hold_s: int, split_end: int
) -> tuple[float, int] | None:
    epochs, bid_o, ask_o, _ = pair_data[pair]
    i = bisect.bisect_left(epochs, entry_after)
    if i >= len(epochs):
        return None
    entry_epoch = int(epochs[i])
    if entry_epoch >= split_end:
        return None
    j = bisect.bisect_left(epochs, entry_epoch + hold_s)
    if j >= len(epochs):
        return None
    exit_epoch = int(epochs[j])
    if exit_epoch >= split_end or exit_epoch <= entry_epoch:
        return None
    if GROSS_MODE:
        entry = (float(ask_o[i]) + float(bid_o[i])) / 2.0
        exit_price = (float(ask_o[j]) + float(bid_o[j])) / 2.0
        raw = (exit_price / entry - 1.0) if side == "LONG" else (entry / exit_price - 1.0)
        return raw * LEVERAGE, exit_epoch
    if side == "LONG":
        entry, exit_price = float(ask_o[i]), float(bid_o[j])
        raw = exit_price / entry - 1.0
    else:
        entry, exit_price = float(bid_o[i]), float(ask_o[j])
        raw = entry / exit_price - 1.0
    return raw * LEVERAGE - COST_STRESS_RETURN, exit_epoch


def _run_candidate(
    lookback_h: int, hold_h: int, variant: str,
    pair_data, hourly, split_from: int, split_end: int,
) -> dict[str, Any]:
    hold_s = hold_h * HOUR
    daily: dict[str, float] = {}
    trades = 0
    busy_until = 0
    hour_epoch = ((split_from + HOUR - 1) // HOUR) * HOUR
    while hour_epoch + hold_s < split_end:
        if hour_epoch < busy_until:
            hour_epoch += HOUR
            continue
        decision = datetime.fromtimestamp(hour_epoch, tz=UTC)
        status = compute_market_status(decision)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= hold_h * 60 + WEEKEND_MARGIN_MIN
        ):
            hour_epoch += HOUR
            continue
        strength = _strength_at(hour_epoch, lookback_h, hourly)
        if strength is None:
            hour_epoch += HOUR
            continue
        ranked = sorted(strength.items(), key=lambda item: -item[1])
        legs: list[tuple[str, str]] = []
        first = _pair_and_side(ranked[0][0], ranked[-1][0])
        if first:
            legs.append(first)
        if variant == "TOP2_BASKET":
            second = _pair_and_side(ranked[1][0], ranked[-2][0])
            if second and (not first or second[0] != first[0]):
                legs.append(second)
        if not legs:
            hour_epoch += HOUR
            continue
        weight = 1.0 / len(legs)
        day = decision.date().isoformat()
        exit_max = 0
        filled = False
        for pair, side in legs:
            result = _execute(pair_data, pair, side, hour_epoch, hold_s, split_end)
            if result is None:
                continue
            ret, exit_epoch = result
            daily[day] = daily.get(day, 0.0) + ret * weight
            trades += 1
            filled = True
            exit_max = max(exit_max, exit_epoch)
        if filled:
            busy_until = exit_max
        hour_epoch += HOUR
    days = sorted(daily)
    compound = 1.0
    worst = 0.0
    negd = 0
    for day in days:
        compound *= 1.0 + daily[day]
        worst = min(worst, daily[day])
        negd += int(daily[day] < 0)
    months = max(1e-9, (split_end - split_from) / (30.44 * 86400))
    return {
        "trades": trades,
        "active_days": len(days),
        "net_nav_return": round(compound - 1.0, 6),
        "monthly_multiple": round(compound ** (1.0 / months), 6),
        "worst_day_nav": round(worst, 6),
        "negative_days": negd,
        "daily": daily,
    }


def _white_reality_check(candidate_dailies: list[dict[str, float]]) -> float:
    all_days = sorted({d for daily in candidate_dailies for d in daily})
    series = [
        [daily.get(day, 0.0) for day in all_days] for daily in candidate_dailies
    ]
    means = [sum(s) / len(s) for s in series]
    observed_max = max(means)
    centered = [
        [value - mean for value in s] for s, mean in zip(series, means)
    ]
    n = len(all_days)
    blocks = list(range(0, n - BOOT_BLOCK_DAYS + 1))
    rng = random.Random(BOOT_SEED)
    exceed = 0
    for _ in range(BOOT_SAMPLES):
        idx: list[int] = []
        while len(idx) < n:
            start = rng.choice(blocks)
            idx.extend(range(start, start + BOOT_BLOCK_DAYS))
        idx = idx[:n]
        boot_max = max(
            sum(s[i] for i in idx) / n for s in centered
        )
        exceed += int(boot_max >= observed_max)
    return exceed / BOOT_SAMPLES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gross", action="store_true", help="mid-to-mid, no spread/cost: signal decomposition mode")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("research output must be clean; refusing stale reuse")
    global GROSS_MODE
    GROSS_MODE = bool(args.gross)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    body_check = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _canonical_sha(body_check):
        raise ValueError("M5 manifest digest is invalid")

    pair_data: dict[str, tuple] = {}
    hourly: dict[str, dict[int, float]] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        data = _load_pair(args.root, pair)
        pair_data[pair] = data
        hourly[pair] = data[3]

    candidates = [
        {"lookback_h": lb, "hold_h": hold, "variant": variant}
        for lb in LOOKBACKS_H
        for hold in HOLDS_H
        for variant in VARIANTS
    ]
    assert len(candidates) == 12

    train_rows = []
    train_dailies = []
    for spec in candidates:
        result = _run_candidate(
            spec["lookback_h"], spec["hold_h"], spec["variant"],
            pair_data, hourly,
            int(TRAIN_FROM.timestamp()), int(TRAIN_TO.timestamp()),
        )
        train_dailies.append(result.pop("daily"))
        train_rows.append({**spec, **result})

    wrc_p = _white_reality_check(train_dailies)
    eligible = [row for row in train_rows if row["net_nav_return"] > 0.0]
    survivor = max(eligible, key=lambda r: r["net_nav_return"]) if eligible else None

    validation = None
    if survivor is not None:
        val = _run_candidate(
            survivor["lookback_h"], survivor["hold_h"], survivor["variant"],
            pair_data, hourly,
            int(TRAIN_TO.timestamp()), int(VAL_TO.timestamp()),
        )
        val.pop("daily")
        validation = val

    body: dict[str, Any] = {
        "contract": "QR_M5_STRENGTH_RESEARCH_V1",
        "schema_version": 1,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "candidate_family": "STRENGTH_12_FIXED_L8_24_120_H4_8_TOP1_TOP2",
        "gross_mode_no_spread": bool(args.gross),
        "accounting": {
            "standard_nav_jpy": NAV_JPY,
            "leverage": LEVERAGE,
            "per_trade_cost_stress_nav": COST_STRESS_RETURN,
        },
        "splits": {
            "train": [TRAIN_FROM.isoformat(), TRAIN_TO.isoformat()],
            "validation": [TRAIN_TO.isoformat(), VAL_TO.isoformat()],
            "test_untouched": [VAL_TO.isoformat(), TEST_TO.isoformat()],
        },
        "train_rows": train_rows,
        "white_reality_check_p": round(wrc_p, 6),
        "train_survivor": survivor,
        "validation_replication": validation,
        "test_opened": False,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "research_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(
        json.dumps(
            {
                "status": "M5_STRENGTH_SEALED",
                "wrc_p": round(wrc_p, 4),
                "survivor": survivor,
                "validation": validation,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
