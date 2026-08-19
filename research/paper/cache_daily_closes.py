"""ファクター分解に要る日足終値をキャッシュする（M1コーパスから再構成）。

`quiet_gate_cache.json` は戦略の日次損益と因果指標しか持っていない。
共通ファクター（USD強弱・JPY強弱…）を作るには **全ペアの日次リターン** が要るので、
日足終値だけを別キャッシュに落とす。日境界は既存コードと同じ UTC。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import load  # noqa: E402
from trend_gate import PAIRS, daily_bars  # noqa: E402

OUT = os.path.join(HERE, "daily_close_cache.json")


def main():
    closes = {}
    for pair in PAIRS:
        print(f"{pair} ...", file=sys.stderr)
        ts, h, l, c, sp = load(pair)
        if len(c) < 200000:
            continue
        days, O, Hh, Ll, C = daily_bars(ts, h, l, c)
        closes[pair] = dict(zip(days, C))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(closes, fh)
    print(f"wrote {OUT}: {len(closes)} pairs", file=sys.stderr)


if __name__ == "__main__":
    main()
