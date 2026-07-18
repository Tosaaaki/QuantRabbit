#!/usr/bin/env python3
"""Discovery batch 2: chase the four unexplained anomalies in sealed results.

Anomalies mined from tonight's sealed artifacts: VALIDATION out-earns TRAIN
2.6x; LONG/SHORT dominance fully REVERSES between windows (killing any
static side rule — the operator's no-direction-bias memory, proven); the
24h-hold extension loses TRAIN yet wins VALIDATION by +50%; the SQUEEZE
cells carry the highest per-trade means but the fewest trades.  Unifying
hypothesis: hold length and size should be REGIME-CONDITIONAL.

Declared arms (batch multiplicity: 8 prior + 4 here = 12, Bonferroni):
  A REGIME_COND_HOLD  — hold 24h only when decision-time regime measures
                        TREND (causal classifier), else the base 12h exit
  B DROP_TRAIN_NEG_PAIRS — exclude pairs with negative TRAIN net (TRAIN-set)
  C CELL_BUDGET_TILT  — pips x1.5 on SQUEEZE-measured trades, x0.75 others
  D A_PLUS_B          — the combination
Selection: TRAIN net >= baseline AND worst day not deeper; VALIDATION rows
disclosed for every arm.  Future window stays the final arbiter.
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
EXTEND_HOLD_SECONDS = 24 * 3600
ARMS = ("BASELINE", "A_REGIME_COND_HOLD", "B_DROP_TRAIN_NEG_PAIRS", "C_CELL_BUDGET_TILT", "D_A_PLUS_B")


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _extend_pips(series, row: TradeOutcome, factor: float, window_end: int) -> float:
    boundary = int(row.entry_utc.timestamp()) + EXTEND_HOLD_SECONDS
    if boundary >= window_end:
        return row.realized_pips
    position = bisect.bisect_left(series.s5_epochs, boundary)
    if position >= len(series.s5_epochs) or int(series.s5_epochs[position]) >= window_end:
        return row.realized_pips
    bid = float(series.bid_opens[position])
    ask = float(series.ask_opens[position])
    if row.side == "LONG":
        return (bid - row.entry_ask) * factor
    return (row.entry_bid - ask) * factor


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

    windows: dict[str, list[TradeOutcome]] = {}
    for label, opened_from, opened_to in (
        ("TRAIN", TRAIN_FROM, TRAIN_TO),
        ("VALIDATION", TRAIN_TO, VALIDATION_TO),
    ):
        windows[label] = list(
            evaluate_spec(
                series, spec=spec, opened_from_utc=opened_from,
                opened_to_utc=opened_to, policy=policy,
            )
        )
    window_end = {"TRAIN": int(TRAIN_TO.timestamp()), "VALIDATION": int(VALIDATION_TO.timestamp())}

    regimes: dict[tuple[str, int], str | None] = {}
    for label, rows in windows.items():
        for i, row in enumerate(rows):
            candles = [c for c in minute_index[row.pair] if c["time"] < row.decision_utc]
            if len(candles) < VOL_HISTORY_WINDOW:
                regimes[(label, i)] = None
                continue
            try:
                c = classify_regime(candles[-VOL_HISTORY_WINDOW:], as_of_utc=row.decision_utc)
                regimes[(label, i)] = c["regime"]
            except RegimeClassifierError:
                regimes[(label, i)] = None

    # TRAIN-set pair exclusion (arm B): pairs with negative TRAIN net.
    train_pair_net: dict[str, float] = {}
    for row in windows["TRAIN"]:
        train_pair_net[row.pair] = train_pair_net.get(row.pair, 0.0) + row.realized_pips
    dropped_pairs = sorted(p for p, v in train_pair_net.items() if v < 0.0)

    def arm_pips(label: str, i: int, row: TradeOutcome, arm: str) -> float | None:
        """None means the trade is excluded under this arm."""

        factor = float(instrument_pip_factor(row.pair))
        regime = regimes.get((label, i))
        if arm in ("B_DROP_TRAIN_NEG_PAIRS", "D_A_PLUS_B") and row.pair in dropped_pairs:
            return None
        pips = row.realized_pips
        if arm in ("A_REGIME_COND_HOLD", "D_A_PLUS_B") and regime == "TREND":
            pips = _extend_pips(series[row.pair], row, factor, window_end[label])
        if arm == "C_CELL_BUDGET_TILT":
            pips = pips * (1.5 if regime == "SQUEEZE" else 0.75)
        return pips

    rows_out = []
    base_train = None
    for arm in ARMS:
        stats = {}
        for label, rows in windows.items():
            adjusted = []
            for i, row in enumerate(rows):
                pips = arm_pips(label, i, row, arm)
                if pips is None:
                    continue
                adjusted.append(
                    TradeOutcome(
                        **{
                            **{f: getattr(row, f) for f in row.__dataclass_fields__},
                            "realized_pips": pips,
                        }
                    )
                )
            stopped, _ = apply_causal_daily_stop(adjusted, stop_pips=STOP_PIPS)
            stats[label] = negative_day_stats(daily_net_pips(stopped))
        if arm == "BASELINE":
            base_train = stats["TRAIN"]
        selected = (
            arm != "BASELINE"
            and stats["TRAIN"]["net_pips"] >= base_train["net_pips"]
            and stats["TRAIN"]["worst_day_pips"] >= base_train["worst_day_pips"]
        )
        rows_out.append(
            {"arm": arm, "train": stats["TRAIN"], "validation": stats["VALIDATION"],
             "train_selected_candidate": selected}
        )

    body: dict[str, Any] = {
        "contract": "QR_DISCOVERY_BATCH_2_V1",
        "schema_version": 1,
        "lock_sha256": lock["lock_sha256"],
        "anomalies_chased": [
            "VAL_OUTEARNS_TRAIN_2_6X",
            "SIDE_DOMINANCE_REVERSES_BETWEEN_WINDOWS",
            "HOLD_EXTENSION_LOSES_TRAIN_WINS_VAL",
            "SQUEEZE_CELLS_HIGHEST_MEAN_FEWEST_TRADES",
        ],
        "declared_arms": list(ARMS),
        "dropped_pairs_train_set": dropped_pairs,
        "selection_rule": "TRAIN_NET_GE_BASELINE_AND_WORST_DAY_NOT_DEEPER",
        "multiplicity_note": "cumulative 12 declared tests tonight; Bonferroni applies",
        "rows": rows_out,
        "future_window_is_final_arbiter": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "batch_sha256": _canonical_sha(body)}
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
                "status": "DISCOVERY_BATCH_2_SEALED",
                "dropped_pairs": dropped_pairs,
                "rows": [
                    {
                        "arm": r["arm"],
                        "train_net": r["train"]["net_pips"],
                        "train_worst": r["train"]["worst_day_pips"],
                        "val_net": r["validation"]["net_pips"],
                        "val_negdays": r["validation"]["negative_days"],
                        "selected": r["train_selected_candidate"],
                    }
                    for r in rows_out
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
