#!/usr/bin/env python3
"""Batch 3: state-conditional tactics — the operator's flexible-trading thesis.

The overlay sweep tested every tactic UNCONDITIONALLY; the operator's
critique is that a pro switches tactics by market state.  Declared arms:

  N_RANGE   situational nanpin: -20p x1 add ONLY when the decision-time
            regime measures RANGE or SQUEEZE (never fight a trend)
  H_TREND   momentum harvest: when regime measures TREND with HIGH vol,
            size x1.5 and take profit at +25p (quick kill); others x0.9
            (budget-neutral tilt)
  COMBO     both together

Money accounting: nanpin arms reserve margin by TRUE concurrency measured
on TRAIN (base leverage set there, replicated on VALIDATION).  Selection:
TRAIN monthly multiple >= baseline AND worst day not deeper; all rows
disclosed.  Cumulative declared tests tonight: 16 (Bonferroni).  Future
window is the final arbiter.  Shadow only.
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
from quant_rabbit.daily_loss_overlay import apply_causal_daily_stop
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
ADD_STEP_PIPS = 20.0
HARVEST_PIPS = 25.0
HARVEST_SIZE = 1.5
OTHER_SIZE = 0.9
PEAK_USAGE_CAP = 0.92
LEVERAGE_CAP = 25.0
SLOTS = 12
TRADING_DAYS = 22
ARMS = ("BASELINE", "N_RANGE", "H_TREND", "COMBO")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _harvest_pips(series, row: TradeOutcome, factor: float) -> float:
    """Full exit at +HARVEST_PIPS if touched before the base exit."""

    take = HARVEST_PIPS / factor
    entry_epoch = int(row.entry_utc.timestamp())
    exit_epoch = int(row.exit_utc.timestamp())
    start = bisect.bisect_right(series.s5_epochs, entry_epoch)
    for index in range(start, len(series.s5_epochs)):
        epoch = int(series.s5_epochs[index])
        if epoch > exit_epoch:
            break
        bid = float(series.bid_opens[index])
        ask = float(series.ask_opens[index])
        if row.side == "LONG" and bid >= row.entry_ask + take:
            return HARVEST_PIPS
        if row.side == "SHORT" and ask <= row.entry_bid - take:
            return HARVEST_PIPS
    return row.realized_pips


def _peak_units(rows_meta) -> int:
    events: list[tuple[int, int]] = []
    for meta in rows_meta:
        events.append((meta["entry"], 1))
        events.append((meta["exit"], -1))
        for epoch in meta["adds"]:
            events.append((epoch, 1))
            events.append((meta["exit"], -1))
    events.sort()
    level = peak = 0
    for _, delta in events:
        level += delta
        peak = max(peak, level)
    return max(peak, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
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

    data: dict[str, list[dict[str, Any]]] = {}
    for label, opened_from, opened_to in (
        ("TRAIN", TRAIN_FROM, TRAIN_TO),
        ("VALIDATION", TRAIN_TO, VALIDATION_TO),
    ):
        outcomes = evaluate_spec(
            series, spec=spec, opened_from_utc=opened_from, opened_to_utc=opened_to,
            policy=policy,
        )
        stopped, _ = apply_causal_daily_stop(outcomes, stop_pips=STOP_PIPS)
        rows = []
        for row in stopped:
            candles = [c for c in minute_index[row.pair] if c["time"] < row.decision_utc]
            regime = vol = None
            if len(candles) >= VOL_HISTORY_WINDOW:
                try:
                    c = classify_regime(candles[-VOL_HISTORY_WINDOW:], as_of_utc=row.decision_utc)
                    regime, vol = c["regime"], c["vol_state"]
                except RegimeClassifierError:
                    pass
            factor = float(instrument_pip_factor(row.pair))
            nanpin_adds: list[int] = []
            nanpin_total = row.realized_pips
            if regime in ("RANGE", "SQUEEZE"):
                try:
                    ladder = resolve_addon_ladder(
                        series[row.pair], row, mode="NANPIN",
                        step_pips=ADD_STEP_PIPS, max_adds=1, pip_factor=factor,
                    )
                    nanpin_adds = [int(e) for e in ladder["add_fill_epochs"]]
                    nanpin_total = float(ladder["blended_total_pips"])
                except Exception:
                    pass
            rows.append(
                {
                    "day": row.decision_utc.date(),
                    "entry": int(row.entry_utc.timestamp()),
                    "exit": int(row.exit_utc.timestamp()),
                    "regime": regime,
                    "vol": vol,
                    "base_pips": row.realized_pips,
                    "nanpin_total_pips": nanpin_total,
                    "adds": nanpin_adds,
                    "harvest_pips": _harvest_pips(series[row.pair], row, factor)
                    if regime == "TREND" and vol == "HIGH"
                    else row.realized_pips,
                }
            )
        data[label] = rows

    # TRUE-concurrency base leverage for arms with adds, set on TRAIN.
    train_peak_with_adds = _peak_units(data["TRAIN"])
    train_peak_base = _peak_units(
        [{**m, "adds": []} for m in data["TRAIN"]]
    )
    lev_with_adds = min(
        LEVERAGE_CAP, PEAK_USAGE_CAP * LEVERAGE_CAP * SLOTS / train_peak_with_adds
    )
    lev_base = min(
        LEVERAGE_CAP, PEAK_USAGE_CAP * LEVERAGE_CAP * SLOTS / train_peak_base
    )

    def arm_daily(rows, arm: str) -> tuple[dict, float]:
        lev = lev_with_adds if arm in ("N_RANGE", "COMBO") else lev_base
        per_pip = 0.0001 * lev / SLOTS
        daily: dict = {}
        for meta in rows:
            if arm == "BASELINE":
                pips = meta["base_pips"]
            elif arm == "N_RANGE":
                pips = meta["nanpin_total_pips"]
            elif arm == "H_TREND":
                tilt = HARVEST_SIZE if (meta["regime"] == "TREND" and meta["vol"] == "HIGH") else OTHER_SIZE
                pips = meta["harvest_pips"] * tilt
            else:  # COMBO
                if meta["regime"] == "TREND" and meta["vol"] == "HIGH":
                    pips = meta["harvest_pips"] * HARVEST_SIZE
                elif meta["regime"] in ("RANGE", "SQUEEZE"):
                    pips = meta["nanpin_total_pips"]
                else:
                    pips = meta["base_pips"] * OTHER_SIZE
            daily[meta["day"]] = daily.get(meta["day"], 0.0) + pips * per_pip
        return daily, lev

    rows_out = []
    base_train_stats = None
    for arm in ARMS:
        stats = {}
        for label in ("TRAIN", "VALIDATION"):
            daily, lev = arm_daily(data[label], arm)
            days = sorted(daily)
            compound = 1.0
            worst = 0.0
            negd = 0
            for day in days:
                compound *= 1.0 + daily[day]
                worst = min(worst, daily[day])
                negd += int(daily[day] < 0)
            stats[label] = {
                "monthly_multiple": round(
                    (compound ** (1.0 / len(days))) ** TRADING_DAYS, 6
                )
                if days
                else 1.0,
                "worst_day_nav": round(worst, 6),
                "negative_days": negd,
                "active_days": len(days),
                "leverage_used": round(lev, 4),
            }
        if arm == "BASELINE":
            base_train_stats = stats["TRAIN"]
        selected = (
            arm != "BASELINE"
            and stats["TRAIN"]["monthly_multiple"] >= base_train_stats["monthly_multiple"]
            and stats["TRAIN"]["worst_day_nav"] >= base_train_stats["worst_day_nav"]
        )
        rows_out.append({"arm": arm, **{k.lower(): v for k, v in stats.items()},
                         "train_selected_candidate": selected})

    body: dict[str, Any] = {
        "contract": "QR_STATE_CONDITIONAL_TACTICS_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "operator_thesis": "tactics switch by measured market state, not fixed rules",
        "declared_arms": list(ARMS),
        "train_peak_units": {"with_adds": train_peak_with_adds, "base": train_peak_base},
        "selection_rule": "TRAIN_MULTIPLE_GE_BASELINE_AND_WORST_DAY_NOT_DEEPER",
        "multiplicity_note": "cumulative 16 declared tests tonight; Bonferroni applies",
        "rows": rows_out,
        "future_window_is_final_arbiter": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "tactics_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(json.dumps({"status": "SEALED", "rows": rows_out}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
