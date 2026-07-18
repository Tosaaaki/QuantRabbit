#!/usr/bin/env python3
"""Lane E1: the operator's proven 2025 profile, mechanized. Single candidate.

Derived mechanically from the sealed 2025 manual history (+266,816 JPY in
two months): USD_JPY only, London-morning session (07:00-10:00 UTC),
with-trend direction only (24h trend filter), ~30-minute micro-breakout
scalps.  Declared rule, fixed BEFORE testing:

  pair USD_JPY; decisions at each closed M5 bar in 07:00-10:00 UTC;
  trend filter: mid close now vs 24h ago (up -> LONG only, down -> SHORT);
  trigger: M5 mid close breaks the previous 4 bars' extreme in trend
  direction; enter next M5 open (ask/bid); TP +10 pips / SL -7 pips
  resolved on M5 bid/ask OHLC pessimistically (both touched in one bar =
  SL first); 2h time-stop at open; one position at a time; weekend gate.

Money accounting: 200k JPY, 10x single position, per-trade cost stress.
Splits: TRAIN 2020-2023, VAL 2024, GROUND-TRUTH window 2025-05-15..07-15
compared against the operator's actual +266,816, TEST 2026 untouched.
No selection, no grid — one declared candidate.  Shadow only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status

UTC = timezone.utc
PAIR = "USD_JPY"
PIP = 0.01
SESSION_UTC = (7, 10)  # London morning [07:00, 10:00)
TREND_LOOKBACK_S = 24 * 3600
BREAK_BARS = 4
TP_PIPS = 10.0
SL_PIPS = 7.0
TIME_STOP_S = 2 * 3600
NAV_JPY = 200_000.0
LEVERAGE = 10.0
COST_STRESS_RETURN = 0.00005 * LEVERAGE
WEEKEND_MARGIN_MIN = 5
SPLITS = {
    "TRAIN_2020_2023": ("2020-01-02", "2024-01-01"),
    "VAL_2024": ("2024-01-01", "2025-01-01"),
    "GROUND_TRUTH_2025_MAY_JUL": ("2025-05-15", "2025-07-15"),
    "REST_2025": ("2025-01-01", "2025-05-15"),
}
OPERATOR_GROUND_TRUTH_JPY = 266_815.9


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(root: Path):
    rows = []
    for shard_file in sorted(root.glob(f"*/{PAIR}/{PAIR}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(
                    datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp()
                )
                rows.append(
                    (
                        epoch,
                        float(row["bid"]["o"]), float(row["bid"]["h"]),
                        float(row["bid"]["l"]), float(row["bid"]["c"]),
                        float(row["ask"]["o"]), float(row["ask"]["h"]),
                        float(row["ask"]["l"]), float(row["ask"]["c"]),
                    )
                )
    rows.sort()
    return rows


def _resolve_exit(rows, k: int, side: str, entry: float) -> tuple[float, int]:
    """Walk bars from entry bar k; pessimistic same-bar TP/SL (SL first)."""

    if side == "LONG":
        tp = entry + TP_PIPS * PIP
        sl = entry - SL_PIPS * PIP
    else:
        tp = entry - TP_PIPS * PIP
        sl = entry + SL_PIPS * PIP
    entry_epoch = rows[k][0]
    for m in range(k, len(rows)):
        epoch, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c = rows[m]
        if epoch >= entry_epoch + TIME_STOP_S:
            exit_price = b_o if side == "LONG" else a_o
            pips = (exit_price - entry) / PIP if side == "LONG" else (entry - exit_price) / PIP
            return pips, epoch
        if side == "LONG":
            sl_hit = b_l <= sl
            tp_hit = b_h >= tp
        else:
            sl_hit = a_h >= sl
            tp_hit = a_l <= tp
        if sl_hit:  # pessimistic: SL first whenever touched
            return -SL_PIPS, epoch
        if tp_hit and epoch > entry_epoch:
            return TP_PIPS, epoch
    return 0.0, rows[-1][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    body_check = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _canonical_sha(body_check):
        raise ValueError("M5 manifest digest is invalid")

    rows = _load(args.root)
    epochs = [r[0] for r in rows]
    mid_close = {r[0] + 300: (r[4] + r[8]) / 2.0 for r in rows}

    trades: list[tuple[int, float, float]] = []  # (decision_epoch, pips, nav)
    busy_until = 0
    for k in range(BREAK_BARS + 1, len(rows) - 1):
        epoch = rows[k][0]
        close_epoch = epoch + 300
        if epoch < busy_until:
            continue
        hour = (close_epoch // 3600) % 24
        if not (SESSION_UTC[0] <= hour < SESSION_UTC[1]):
            continue
        now = mid_close.get(close_epoch)
        past = mid_close.get(close_epoch - TREND_LOOKBACK_S)
        if now is None or past is None:
            continue
        trend = "LONG" if now > past else "SHORT"
        window = rows[k - BREAK_BARS: k]
        prev_mid_high = max((r[4] + r[8]) / 2.0 for r in window)
        prev_mid_low = min((r[4] + r[8]) / 2.0 for r in window)
        bar_mid_close = (rows[k][4] + rows[k][8]) / 2.0
        triggered = (
            trend == "LONG" and bar_mid_close > prev_mid_high
        ) or (trend == "SHORT" and bar_mid_close < prev_mid_low)
        if not triggered:
            continue
        decision = datetime.fromtimestamp(close_epoch, tz=UTC)
        status = compute_market_status(decision)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= TIME_STOP_S / 60 + WEEKEND_MARGIN_MIN
        ):
            continue
        entry_bar = k + 1
        if rows[entry_bar][0] != epoch + 300:
            continue  # gap: no immediate next bar; skip (no chase)
        entry = rows[entry_bar][5] if trend == "LONG" else rows[entry_bar][1]
        pips, exit_epoch = _resolve_exit(rows, entry_bar, trend, entry)
        nav = (pips * PIP / entry) * LEVERAGE - COST_STRESS_RETURN
        trades.append((close_epoch, pips, nav))
        busy_until = exit_epoch

    def split_stats(from_s: str, to_s: str) -> dict[str, Any]:
        f = int(datetime.fromisoformat(from_s + "T00:00:00+00:00").timestamp())
        t = int(datetime.fromisoformat(to_s + "T00:00:00+00:00").timestamp())
        daily: dict[str, float] = {}
        pips_sum = 0.0
        n = 0
        for epoch, pips, nav in trades:
            if f <= epoch < t:
                day = datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()
                daily[day] = daily.get(day, 0.0) + nav
                pips_sum += pips
                n += 1
        compound = 1.0
        worst = 0.0
        negd = 0
        for day in sorted(daily):
            compound *= 1.0 + daily[day]
            worst = min(worst, daily[day])
            negd += int(daily[day] < 0)
        months = max(1e-9, (t - f) / (30.44 * 86400))
        return {
            "trades": n,
            "net_pips": round(pips_sum, 1),
            "nav_multiple": round(compound, 6),
            "monthly_multiple": round(compound ** (1.0 / months), 6),
            "profit_jpy_from_200k": round(NAV_JPY * (compound - 1.0)),
            "active_days": len(daily),
            "negative_days": negd,
            "worst_day_nav": round(worst, 6),
        }

    results = {name: split_stats(f, t) for name, (f, t) in SPLITS.items()}
    body: dict[str, Any] = {
        "contract": "QR_LANE_E1_OPERATOR_PRECEDENT_V1",
        "schema_version": 1,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "declared_rule": (
            "USDJPY LondonAM 07-10UTC, 24h trend side only, 4-bar micro-break, "
            "TP10/SL7 pessimistic same-bar SL-first, 2h time stop, one position"
        ),
        "derived_from": "manual_history_2025_mining (+266,816 JPY, 411 trades)",
        "single_candidate_no_selection": True,
        "accounting": {"standard_nav_jpy": NAV_JPY, "leverage": LEVERAGE},
        "splits": results,
        "operator_ground_truth_jpy": OPERATOR_GROUND_TRUTH_JPY,
        "ground_truth_comparison": {
            "machine_jpy": results["GROUND_TRUTH_2025_MAY_JUL"]["profit_jpy_from_200k"],
            "operator_jpy": OPERATOR_GROUND_TRUTH_JPY,
        },
        "test_2026_untouched": True,
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
    print(json.dumps({"status": "LANE_E1_SEALED", "splits": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
