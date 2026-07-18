#!/usr/bin/env python3
"""Declared overlay sweep: eight pre-registered improvements in one pass.

Operator directive: stop testing one idea at a time — declare the whole
grid and measure it.  Eight overlays are pre-registered here (exits:
trailing stop, breakeven arm, partial-take + runner, 24h extension; adds:
bounded pyramid, bounded nanpin; sizing: losing-cell half-size, pair-vol
normalization).  Selection rule, declared before results: an overlay is a
candidate only if TRAIN net >= baseline AND TRAIN worst day is not deeper;
every row's VALIDATION replication is disclosed regardless.  Eight tests
carry a Bonferroni-style multiplicity note; the future window stays the
final arbiter.  Exits obey executable sides (LONG exits at bid, SHORT at
ask).  Shadow only.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.adaptive_exact_s5_profit_engine import (
    TradeOutcome,
    _policy_from_payload,
    _spec_from_payload,
    _validate_lock,
    evaluate_spec,
    prepare_exact_s5_series,
)
from quant_rabbit.addon_ladder_shadow import resolve_addon_ladder
from quant_rabbit.daily_loss_overlay import (
    apply_causal_daily_stop,
    daily_net_pips,
    negative_day_stats,
)
from quant_rabbit.fast_bot_historical_s5 import load_historical_s5_slice
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.regime_classifier_shadow import (
    VOL_HISTORY_WINDOW,
    RegimeClassifierError,
    classify_regime,
)

TRAIN_FROM = datetime(2026, 5, 12, tzinfo=timezone.utc)
TRAIN_TO = datetime(2026, 6, 15, tzinfo=timezone.utc)
VALIDATION_TO = datetime(2026, 6, 28, tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=725)
STOP_PIPS = 50.0
TRAIL_PIPS = 30.0
BREAKEVEN_ARM_PIPS = 20.0
PARTIAL_TAKE_PIPS = 20.0
EXTEND_HOLD_SECONDS = 24 * 3600
ADD_STEP_PIPS = 20.0
LOSING_CELLS = ("RANGE_HIGH", "TREND_LOW")  # from sealed W27 attribution
OVERLAYS = (
    "BASELINE",
    "TRAILING_STOP_30P",
    "BREAKEVEN_AFTER_20P",
    "HALF_TAKE_20P_PLUS_RUNNER",
    "EXTEND_HOLD_24H",
    "PYRAMID_20P_MAX2",
    "NANPIN_20P_MAX1",
    "LOSING_CELL_HALF_SIZE",
    "PAIR_VOL_NORMALIZED",
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _walk(series, row: TradeOutcome):
    """Yield (epoch, bid_open, ask_open) strictly after entry, through exit."""

    entry_epoch = int(row.entry_utc.timestamp())
    exit_epoch = int(row.exit_utc.timestamp())
    start = bisect.bisect_right(series.s5_epochs, entry_epoch)
    for index in range(start, len(series.s5_epochs)):
        epoch = int(series.s5_epochs[index])
        if epoch > exit_epoch:
            break
        yield epoch, float(series.bid_opens[index]), float(series.ask_opens[index])


def _trailing(series, row: TradeOutcome, factor: float) -> float:
    trail = TRAIL_PIPS / factor
    if row.side == "LONG":
        peak = row.entry_ask
        for _, bid, _ask in _walk(series, row):
            peak = max(peak, bid)
            if bid <= peak - trail:
                return (bid - row.entry_ask) * factor
        return row.realized_pips
    trough = row.entry_bid
    for _, _bid, ask in _walk(series, row):
        trough = min(trough, ask)
        if ask >= trough + trail:
            return (row.entry_bid - ask) * factor
    return row.realized_pips


def _breakeven(series, row: TradeOutcome, factor: float) -> float:
    arm = BREAKEVEN_ARM_PIPS / factor
    armed = False
    if row.side == "LONG":
        for _, bid, _ask in _walk(series, row):
            if not armed and bid >= row.entry_ask + arm:
                armed = True
            elif armed and bid <= row.entry_ask:
                return (bid - row.entry_ask) * factor
        return row.realized_pips
    for _, _bid, ask in _walk(series, row):
        if not armed and ask <= row.entry_bid - arm:
            armed = True
        elif armed and ask >= row.entry_bid:
            return (row.entry_bid - ask) * factor
    return row.realized_pips


def _half_take(series, row: TradeOutcome, factor: float) -> float:
    take = PARTIAL_TAKE_PIPS / factor
    if row.side == "LONG":
        for _, bid, _ask in _walk(series, row):
            if bid >= row.entry_ask + take:
                return 0.5 * PARTIAL_TAKE_PIPS + 0.5 * row.realized_pips
        return row.realized_pips
    for _, _bid, ask in _walk(series, row):
        if ask <= row.entry_bid - take:
            return 0.5 * PARTIAL_TAKE_PIPS + 0.5 * row.realized_pips
    return row.realized_pips


def _extend_24h(series, row: TradeOutcome, factor: float, window_end: int) -> float:
    boundary = int(row.entry_utc.timestamp()) + EXTEND_HOLD_SECONDS
    if boundary >= window_end:
        return row.realized_pips  # declared: no extension across the split
    position = bisect.bisect_left(series.s5_epochs, boundary)
    if position >= len(series.s5_epochs) or int(series.s5_epochs[position]) >= window_end:
        return row.realized_pips
    bid = float(series.bid_opens[position])
    ask = float(series.ask_opens[position])
    if row.side == "LONG":
        return (bid - row.entry_ask) * factor
    return (row.entry_bid - ask) * factor


def _addon(series, row: TradeOutcome, factor: float, mode: str, max_adds: int) -> float:
    try:
        result = resolve_addon_ladder(
            series, row, mode=mode, step_pips=ADD_STEP_PIPS, max_adds=max_adds,
            pip_factor=factor,
        )
    except Exception:
        return row.realized_pips
    # Per-unit blending keeps portfolio exposure comparable to baseline.
    return float(result["blended_pips_per_unit"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("sweep output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    _validate_lock(lock)
    spec = _spec_from_payload(lock["spec"])
    policy = _policy_from_payload(lock["evaluation_policy"])

    series = {}
    minute_index: dict[str, list] = {}
    for pair in DEFAULT_TRADER_PAIRS:
        item = load_historical_s5_slice(
            manifest, pair=pair, time_from=TRAIN_FROM - LOOKBACK, time_to=VALIDATION_TO
        )
        prepared = prepare_exact_s5_series(item.candles)
        series[pair] = prepared
        minute_index[pair] = [
            {"time": p.minute_utc, "close": p.mid_close} for p in prepared.points
        ]
        del item
        gc.collect()

    windows = {
        "TRAIN": evaluate_spec(
            series, spec=spec, opened_from_utc=TRAIN_FROM, opened_to_utc=TRAIN_TO,
            policy=policy,
        ),
        "VALIDATION": evaluate_spec(
            series, spec=spec, opened_from_utc=TRAIN_TO, opened_to_utc=VALIDATION_TO,
            policy=policy,
        ),
    }
    window_end = {"TRAIN": int(TRAIN_TO.timestamp()), "VALIDATION": int(VALIDATION_TO.timestamp())}

    # Pair-vol weights declared from TRAIN only: inverse of median |trade pips|.
    by_pair: dict[str, list[float]] = {}
    for row in windows["TRAIN"]:
        by_pair.setdefault(row.pair, []).append(abs(row.realized_pips))
    medians = {
        pair: sorted(vals)[len(vals) // 2] for pair, vals in by_pair.items() if vals
    }
    mean_median = sum(medians.values()) / len(medians)
    vol_weight = {
        pair: min(3.0, mean_median / value) if value > 0 else 1.0
        for pair, value in medians.items()
    }

    # Regime cell per trade (for LOSING_CELL_HALF_SIZE), measured causally.
    def cell_of(row: TradeOutcome) -> str | None:
        candles = [c for c in minute_index[row.pair] if c["time"] < row.decision_utc]
        if len(candles) < VOL_HISTORY_WINDOW:
            return None
        try:
            c = classify_regime(candles[-VOL_HISTORY_WINDOW - 5:], as_of_utc=row.decision_utc)
        except RegimeClassifierError:
            return None
        return f"{c['regime']}_{c['vol_state']}"

    cells: dict[tuple[str, int], str | None] = {}
    for label, rows in windows.items():
        for i, row in enumerate(rows):
            cells[(label, i)] = cell_of(row)

    def overlay_pips(label: str, i: int, row: TradeOutcome, overlay: str) -> float:
        factor = float(instrument_pip_factor(row.pair))
        s = series[row.pair]
        if overlay == "BASELINE":
            return row.realized_pips
        if overlay == "TRAILING_STOP_30P":
            return _trailing(s, row, factor)
        if overlay == "BREAKEVEN_AFTER_20P":
            return _breakeven(s, row, factor)
        if overlay == "HALF_TAKE_20P_PLUS_RUNNER":
            return _half_take(s, row, factor)
        if overlay == "EXTEND_HOLD_24H":
            return _extend_24h(s, row, factor, window_end[label])
        if overlay == "PYRAMID_20P_MAX2":
            return _addon(s, row, factor, "PYRAMID", 2)
        if overlay == "NANPIN_20P_MAX1":
            return _addon(s, row, factor, "NANPIN", 1)
        if overlay == "LOSING_CELL_HALF_SIZE":
            scale = 0.5 if cells.get((label, i)) in LOSING_CELLS else 1.0
            return row.realized_pips * scale
        if overlay == "PAIR_VOL_NORMALIZED":
            return row.realized_pips * vol_weight.get(row.pair, 1.0)
        raise ValueError(overlay)

    results: dict[str, dict[str, Any]] = {}
    for overlay in OVERLAYS:
        results[overlay] = {}
        for label, rows in windows.items():
            adjusted = [
                TradeOutcome(
                    **{
                        **{f: getattr(row, f) for f in row.__dataclass_fields__},
                        "realized_pips": overlay_pips(label, i, row, overlay),
                    }
                )
                for i, row in enumerate(rows)
            ]
            stopped, _ = apply_causal_daily_stop(adjusted, stop_pips=STOP_PIPS)
            results[overlay][label] = negative_day_stats(daily_net_pips(stopped))

    base_train = results["BASELINE"]["TRAIN"]
    table = []
    for overlay in OVERLAYS:
        train = results[overlay]["TRAIN"]
        val = results[overlay]["VALIDATION"]
        selected = (
            overlay != "BASELINE"
            and train["net_pips"] >= base_train["net_pips"]
            and train["worst_day_pips"] >= base_train["worst_day_pips"]
        )
        table.append(
            {
                "overlay": overlay,
                "train": train,
                "validation": val,
                "train_selected_candidate": selected,
            }
        )

    body: dict[str, Any] = {
        "contract": "QR_OVERLAY_SWEEP_LAB_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "declared_overlays": list(OVERLAYS),
        "selection_rule": (
            "TRAIN_NET_GE_BASELINE_AND_WORST_DAY_NOT_DEEPER_DECLARED_BEFORE_RESULTS"
        ),
        "multiplicity_note": "8 tests; Bonferroni discount applies to any VAL claim",
        "pair_vol_weights_from_train_only": True,
        "losing_cells_from_sealed_attribution": list(LOSING_CELLS),
        "rows": table,
        "future_window_is_final_arbiter": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "sweep_sha256": _canonical_sha(body)}
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
                "status": "SWEEP_SEALED",
                "rows": [
                    {
                        "overlay": r["overlay"],
                        "train_net": r["train"]["net_pips"],
                        "train_worst": r["train"]["worst_day_pips"],
                        "val_net": r["validation"]["net_pips"],
                        "val_negdays": r["validation"]["negative_days"],
                        "selected": r["train_selected_candidate"],
                    }
                    for r in table
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
