#!/usr/bin/env python3
"""Build anonymized blind-read scenarios for the AI-discretion pilot.

Samples decision points from the sealed M5 corpus, extracts the trailing
structure (120 H1 mid closes + 36 M5 mid OHLC bars), normalizes prices to
100 at the window start, and STRIPS dates, pair names, and absolute levels
so a reader cannot recall what happened next — hindsight is structurally
impossible.  The forward answers (2h and 12h mid moves) go to a separate
sealed answer key that readers never see.
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
SEED = 20260718
H1_BARS = 120
M5_BARS = 36
FWD_SHORT_S = 2 * 3600
FWD_LONG_S = 12 * 3600


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_pair(root: Path, pair: str):
    rows = []
    for shard_file in sorted(root.glob(f"*/{pair}/{pair}_M5_BA_*.jsonl.gz")):
        with gzip.open(shard_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                epoch = int(datetime.fromisoformat(row["time"][:19] + "+00:00").timestamp())
                mid_o = (float(row["bid"]["o"]) + float(row["ask"]["o"])) / 2.0
                mid_h = (float(row["bid"]["h"]) + float(row["ask"]["h"])) / 2.0
                mid_l = (float(row["bid"]["l"]) + float(row["ask"]["l"])) / 2.0
                mid_c = (float(row["bid"]["c"]) + float(row["ask"]["c"])) / 2.0
                rows.append((epoch, mid_o, mid_h, mid_l, mid_c))
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
    while len(scenarios) < N_SCENARIOS and attempts < 5000:
        attempts += 1
        pair = rng.choice(PAIRS)
        rows = data[pair]
        k = rng.randrange(H1_BARS * 12 + M5_BARS + 10, len(rows) - FWD_LONG_S // 300 - 10)
        epoch = rows[k][0]
        stamp = datetime.fromtimestamp(epoch, tz=UTC)
        if not (FROM <= stamp < TO):
            continue
        status = compute_market_status(stamp)
        if not status.is_fx_open or status.minutes_to_next_close is None or (
            status.minutes_to_next_close <= FWD_LONG_S / 60 + 30
        ):
            continue
        # H1 closes: the last 120 TRADED hours (skip market-closed hours),
        # newest-last, walking back up to 400 calendar hours.
        h1_rev: list[float] = []
        hour_anchor = (epoch // 3600) * 3600
        for h in range(1, 400):
            if len(h1_rev) >= H1_BARS:
                break
            target = hour_anchor - h * 3600 + 3600 - 300
            i = index[pair].get(target)
            if i is None:
                for back in range(1, 12):
                    j = index[pair].get(target - back * 300)
                    if j is not None:
                        i = j
                        break
            if i is not None:
                h1_rev.append(rows[i][4])
        if len(h1_rev) < H1_BARS:
            continue
        h1_closes = list(reversed(h1_rev))
        m5_window = rows[k - M5_BARS + 1 : k + 1]
        base = h1_closes[0]
        if base <= 0:
            continue
        norm = lambda x: round(x / base * 100.0, 4)
        # Forward answers from the NEXT M5 open (executable) to horizon opens.
        entry_i = k + 1
        if entry_i >= len(rows):
            continue
        entry_epoch = rows[entry_i][0]
        def fwd(seconds: int):
            target = entry_epoch + seconds
            j = entry_i
            while j < len(rows) and rows[j][0] < target:
                j += 1
            return rows[j][1] if j < len(rows) else None
        entry_price = rows[entry_i][1]
        p2 = fwd(FWD_SHORT_S)
        p12 = fwd(FWD_LONG_S)
        if p2 is None or p12 is None:
            continue
        sid = f"S{len(scenarios)+1:02d}"
        scenarios.append(
            {
                "id": sid,
                "h1_closes_norm": [norm(c) for c in h1_closes],
                "m5_ohlc_norm": [
                    [norm(o), norm(h), norm(l), norm(c)] for _, o, h, l, c in m5_window
                ],
            }
        )
        answers.append(
            {
                "id": sid,
                "pair": pair,
                "epoch": epoch,
                "entry_norm": norm(entry_price),
                "fwd_2h_norm": norm(p2),
                "fwd_12h_norm": norm(p12),
            }
        )

    if len(scenarios) < N_SCENARIOS:
        raise ValueError(f"only {len(scenarios)} scenarios sampled")

    for path, payload in (
        (args.scenarios_out, {"contract": "QR_BLIND_READ_SCENARIOS_V1", "seed": SEED, "scenarios": scenarios}),
        (args.answer_key_out, {"contract": "QR_BLIND_READ_ANSWER_KEY_V1", "seed": SEED, "answers": answers}),
    ):
        sealed = {**payload, "sha256": _canonical_sha(payload)}
        text = json.dumps(sealed, ensure_ascii=False, sort_keys=True) + "\n"
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    print(json.dumps({"status": "SCENARIOS_SEALED", "count": len(scenarios)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
