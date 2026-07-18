#!/usr/bin/env python3
"""Blind EXIT-judgment scenarios: can discretion beat the blunt time-stop?

Samples mechanical burst entries (same declared trigger as the MomentumBurst
replica: 24h trend side, 3-bar M5 break, next-open entry, TP +3 pips
pessimistic) that are still open 60 minutes later — the exact moment the
mechanical time-stop acted in the protected arm.  Each scenario shows the
anonymized trailing structure (120 H1 closes + 36 M5 bars, normalized to
100, dates/pairs/levels stripped) plus the entry marker and current
unrealized pips, and asks one binary decision: CUT now or HOLD (keep the
TP, hard exit 4h later if unfilled).  The sealed answer key carries both
branches' forward pips so the reader policy can be scored against
ALWAYS_CUT (the mechanical baseline), ALWAYS_HOLD, and the oracle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status

UTC = timezone.utc
PAIRS = ("EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF", "USD_CAD", "EUR_JPY", "GBP_JPY")
FROM = datetime(2020, 3, 1, tzinfo=UTC)
TO = datetime(2025, 12, 1, tzinfo=UTC)
N_SCENARIOS = 40
SEED = 20260719
H1_BARS = 120
M5_BARS = 36
TREND_LOOKBACK_S = 24 * 3600
BREAK_BARS = 3
TP_PIPS = 3.0
DECISION_DELAY_S = 60 * 60
HOLD_HORIZON_S = 4 * 3600


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _load_pair(root: Path, pair: str):
    rows = []
    for shard_file in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp())
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenarios-out", type=Path, required=True)
    parser.add_argument("--answer-key-out", type=Path, required=True)
    args = parser.parse_args()

    data = {pair: _load_pair(args.root, pair) for pair in PAIRS}
    index = {pair: {r[0]: i for i, r in enumerate(rows)} for pair, rows in data.items()}

    rng = random.Random(SEED)
    scenarios: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    attempts = 0
    while len(scenarios) < N_SCENARIOS and attempts < 40000:
        attempts += 1
        pair = rng.choice(PAIRS)
        rows = data[pair]
        pip = _pip(pair)
        k = rng.randrange(H1_BARS * 12 + M5_BARS + 10, len(rows) - (DECISION_DELAY_S + HOLD_HORIZON_S) // 300 - 20)
        epoch = rows[k][0]
        stamp = datetime.fromtimestamp(epoch, tz=UTC)
        if not (FROM <= stamp < TO):
            continue
        status = compute_market_status(stamp)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= (DECISION_DELAY_S + HOLD_HORIZON_S) / 60 + 30
        ):
            continue
        # Mechanical burst trigger at bar k (closed) — same as replica.
        mid_close = (rows[k][4] + rows[k][8]) / 2.0
        close_epoch = epoch + 300
        past_i = index[pair].get(close_epoch - TREND_LOOKBACK_S - 300)
        if past_i is None:
            continue
        past_mid = (data[pair][past_i][4] + data[pair][past_i][8]) / 2.0
        trend = "LONG" if mid_close > past_mid else "SHORT"
        window = rows[k - BREAK_BARS: k]
        mids_h = [(r[2] + r[6]) / 2.0 for r in window]
        mids_l = [(r[3] + r[7]) / 2.0 for r in window]
        triggered = (
            (trend == "LONG" and mid_close > max(mids_h))
            or (trend == "SHORT" and mid_close < min(mids_l))
        )
        if not triggered:
            continue
        if k + 1 >= len(rows) or rows[k + 1][0] != epoch + 300:
            continue
        entry_bar = k + 1
        entry_epoch = rows[entry_bar][0]
        entry = rows[entry_bar][5] if trend == "LONG" else rows[entry_bar][1]
        tp = entry + TP_PIPS * pip if trend == "LONG" else entry - TP_PIPS * pip

        # Walk to the decision point; drop the scenario if TP fills first
        # (pessimistic: no earlier than the bar after entry).
        decision_epoch_target = entry_epoch + DECISION_DELAY_S
        j = entry_bar + 1
        tp_before_decision = False
        while j < len(rows) and rows[j][0] < decision_epoch_target:
            _, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c = rows[j]
            if (trend == "LONG" and b_h >= tp) or (trend == "SHORT" and a_l <= tp):
                tp_before_decision = True
                break
            j += 1
        if tp_before_decision or j >= len(rows):
            continue
        decision_bar = j
        d_epoch = rows[decision_bar][0]
        if d_epoch - entry_epoch > DECISION_DELAY_S + 1800:
            continue  # data gap around the decision point

        # Branch A: CUT at the decision bar open.
        cut_price = rows[decision_bar][1] if trend == "LONG" else rows[decision_bar][5]
        cut_pips = (cut_price - entry) / pip if trend == "LONG" else (entry - cut_price) / pip
        # Branch B: HOLD — TP stays working, hard exit at decision+4h open.
        hold_end = d_epoch + HOLD_HORIZON_S
        m = decision_bar
        hold_pips = None
        while m < len(rows) and rows[m][0] < hold_end:
            _, b_o, b_h, b_l, b_c, a_o, a_h, a_l, a_c = rows[m]
            if m > decision_bar and (
                (trend == "LONG" and b_h >= tp) or (trend == "SHORT" and a_l <= tp)
            ):
                hold_pips = TP_PIPS
                break
            m += 1
        if hold_pips is None:
            if m >= len(rows):
                continue
            end_price = rows[m][1] if trend == "LONG" else rows[m][5]
            hold_pips = (end_price - entry) / pip if trend == "LONG" else (entry - end_price) / pip

        # Anonymized context: last 120 traded-hour H1 closes + last 36 M5 bars.
        h1_rev: list[float] = []
        hour_anchor = (d_epoch // 3600) * 3600
        for h in range(1, 400):
            if len(h1_rev) >= H1_BARS:
                break
            target = hour_anchor - h * 3600 + 3600 - 300
            i = index[pair].get(target)
            if i is None:
                for back in range(1, 12):
                    alt = index[pair].get(target - back * 300)
                    if alt is not None:
                        i = alt
                        break
            if i is not None:
                h1_rev.append((data[pair][i][4] + data[pair][i][8]) / 2.0)
        if len(h1_rev) < H1_BARS:
            continue
        h1_closes = list(reversed(h1_rev))
        m5_window = rows[decision_bar - M5_BARS + 1: decision_bar + 1]
        base = h1_closes[0]
        if base <= 0:
            continue

        def norm(x: float) -> float:
            return round(x / base * 100.0, 4)

        sid = f"X{len(scenarios)+1:02d}"
        scenarios.append(
            {
                "id": sid,
                "side": trend,
                "entry_norm": norm(entry),
                "tp_norm": norm(tp),
                "current_norm": norm(cut_price),
                "unrealized_pips": round(cut_pips, 1),
                "minutes_held": int((d_epoch - entry_epoch) / 60),
                "h1_closes_norm": [norm(c) for c in h1_closes],
                "m5_ohlc_norm": [
                    [norm((r[1] + r[5]) / 2), norm((r[2] + r[6]) / 2),
                     norm((r[3] + r[7]) / 2), norm((r[4] + r[8]) / 2)]
                    for r in m5_window
                ],
            }
        )
        answers.append(
            {
                "id": sid,
                "pair": pair,
                "entry_epoch": entry_epoch,
                "side": trend,
                "cut_pips": round(cut_pips, 2),
                "hold_pips": round(hold_pips, 2),
            }
        )

    if len(scenarios) < N_SCENARIOS:
        raise ValueError(f"only {len(scenarios)} scenarios sampled after {attempts} attempts")

    for path, payload in (
        (args.scenarios_out, {"contract": "QR_BLIND_EXIT_SCENARIOS_V1", "seed": SEED,
                              "decision": "CUT_NOW_or_HOLD(TP working, hard exit +4h)",
                              "scenarios": scenarios}),
        (args.answer_key_out, {"contract": "QR_BLIND_EXIT_ANSWER_KEY_V1", "seed": SEED,
                               "answers": answers}),
    ):
        sealed = {**payload, "sha256": _canonical_sha(payload)}
        text = json.dumps(sealed, ensure_ascii=False, sort_keys=True) + "\n"
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    cut_mean = sum(a["cut_pips"] for a in answers) / len(answers)
    hold_mean = sum(a["hold_pips"] for a in answers) / len(answers)
    print(json.dumps({"status": "EXIT_SCENARIOS_SEALED", "count": len(scenarios),
                      "baseline_always_cut_mean_pips": round(cut_mean, 3),
                      "baseline_always_hold_mean_pips": round(hold_mean, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
