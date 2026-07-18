#!/usr/bin/env python3
"""Per-pair geometry adaptation lab: give every pair its own suit, then
execute the survivors' gates.

Phase 1  screen: default-geometry fade TRAIN (2024-07..2025-07) on the
         newly fetched cross pairs.
Phase 2  adaptation: candidates (phase-1 net > -20k JPY, plus the S5
         screen positives) run a small declared per-pair grid
         (fade_atr x eff_max x tp_atr = 8 configs) on TRAIN; the best
         TRAIN-positive config is selected per pair.
Phase 3  gates: each selected (pair, config) must pass untouched VAL
         (2025-07..2026-07, M1) AND the hardened S5 window (55 days,
         slippage 0.3p + financing 0.8p/day).  Only all-gates survivors
         join the floor.

Multiplicity is explicit: every run is recorded in the scoreboard with
its phase, and phase-2 selection happens ONLY on TRAIN.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
M1_ROOT = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"
S5_ROOT = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history"
TRAIN = ("2024-07-01T00:00:00", "2025-07-01T00:00:00")
VAL = ("2025-07-01T00:00:00", "2026-07-01T00:00:00")
S5W = ("2026-05-10T00:00:00", "2026-07-04T00:00:00")

NEW_PAIRS = ["EUR_GBP", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
             "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "AUD_NZD",
             "AUD_CAD", "AUD_CHF", "NZD_CAD", "NZD_CHF", "CAD_CHF",
             "USD_CHF", "USD_CAD"]
S5_SCREEN_POSITIVE = ["NZD_JPY", "CAD_JPY", "AUD_CHF", "NZD_CAD", "EUR_NZD"]

GRID = [
    {"fade_atr": fa, "eff_max": em, "tp_atr": tp}
    for fa in (1.0, 1.5) for em in (0.15, 0.25) for tp in (2.0, 3.0)
]


def base_cfg(pair: str, geo: dict) -> dict:
    return {
        "signal": "range_fade_limit", "pairs": [pair],
        "tp_atr": geo["tp_atr"], "sl_pips": None, "ceiling_min": 480,
        "max_concurrent": 1, "per_pos_lev": 4.3, "atr_floor_pips": 0.5,
        "fade_atr": geo["fade_atr"], "eff_max": geo["eff_max"],
    }


def run(pair: str, cfg: dict, window: tuple[str, str], tag: str,
        s5: bool = False, hardened: bool = False) -> dict:
    out = Path(os.environ.get("LAB_OUT", "/tmp")) / f"pa_{tag}"
    session = out
    if session.exists():
        subprocess.run(["rm", "-rf", str(session)], check=False)
    (session / "inbox").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DOJO_BOT_CONFIG"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(REPO / "src")
    cmd = [PY, str(REPO / "scripts/run-virtual-market-session.py"),
           "--feed", "replay", "--session-dir", str(session),
           "--pairs", pair, "--from", window[0], "--to", window[1],
           "--bars-per-second", "100000", "--state-every", "100000",
           "--fast-ledger", "--bot-module", str(REPO / "bots/lab_bot.py") + ":Bot"]
    if s5:
        cmd += ["--granularity", "S5", "--bot-bar", "M1", "--corpus-root", S5_ROOT]
    else:
        cmd += ["--corpus-root", M1_ROOT]
    if hardened:
        cmd += ["--slippage-pips", "0.3", "--financing-pips-day", "0.8"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        return {"tag": tag, "error": proc.stderr[-200:]}
    pl = trades = wins = 0.0
    daily: dict[str, float] = {}
    for line in (session / "ledger.jsonl").open():
        rec = json.loads(line)
        pay = rec["payload"] if isinstance(rec["payload"], dict) else {}
        p = pay.get("pl_jpy")
        if p is not None and rec["event"].startswith(("EXIT", "CLOSE", "MARGIN")):
            pl += p; trades += 1; wins += p > 0
            d = (pay.get("quote") or {}).get("ts", "")[:10] or rec["ts_utc"][:10]
            daily[d] = daily.get(d, 0.0) + p
    eq, mult, worst = 200_000.0, 1.0, 0.0
    for d in sorted(daily):
        r = max(daily[d] / max(eq, 1e-9), -1.0)
        eq = max(eq + daily[d], 0.0)
        worst = min(worst, r)
        mult = max(mult * (1 + r), 0.0)
    months = max(len(daily) / 21.7, 1e-9)
    return {"tag": tag, "net_jpy": round(pl), "trades": int(trades),
            "win_rate": round(wins / max(trades, 1), 3),
            "monthly_multiple": round(mult ** (1 / months), 4) if mult > 0 else 0.0,
            "worst_day": round(worst, 4)}


def main() -> int:
    out_root = Path(sys.argv[1])
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ["LAB_OUT"] = str(out_root)
    board: list[dict] = []

    default_geo = {"fade_atr": 1.2, "eff_max": 0.2, "tp_atr": 3.0}
    phase1_pass: list[str] = []
    for pair in NEW_PAIRS:
        row = run(pair, base_cfg(pair, default_geo), TRAIN, f"p1_{pair}")
        row.update({"phase": 1, "pair": pair})
        board.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if (row.get("net_jpy") or -10**9) > -20_000:
            phase1_pass.append(pair)

    candidates = sorted(set(phase1_pass) | set(S5_SCREEN_POSITIVE))
    print(f"phase2 candidates: {candidates}", flush=True)

    selected: dict[str, dict] = {}
    for pair in candidates:
        best = None
        for i, geo in enumerate(GRID):
            row = run(pair, base_cfg(pair, geo), TRAIN, f"p2_{pair}_{i}")
            row.update({"phase": 2, "pair": pair, "geo": geo})
            board.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row.get("net_jpy") is not None and (
                    best is None or row["net_jpy"] > best["net_jpy"]):
                best = row
        if best and best["net_jpy"] > 0:
            selected[pair] = best["geo"]
            print(f"SELECTED {pair}: {best['geo']} (TRAIN +{best['net_jpy']})", flush=True)

    survivors = []
    for pair, geo in selected.items():
        val = run(pair, base_cfg(pair, geo), VAL, f"p3v_{pair}")
        val.update({"phase": 3, "gate": "VAL", "pair": pair, "geo": geo})
        board.append(val)
        print(json.dumps(val, ensure_ascii=False), flush=True)
        if (val.get("net_jpy") or -1) <= 0:
            continue
        s5 = run(pair, base_cfg(pair, geo), S5W, f"p3s_{pair}", s5=True, hardened=True)
        s5.update({"phase": 3, "gate": "S5_HARDENED", "pair": pair, "geo": geo})
        board.append(s5)
        print(json.dumps(s5, ensure_ascii=False), flush=True)
        if (s5.get("net_jpy") or -1) > 0:
            survivors.append({"pair": pair, "geo": geo,
                              "val_net": val["net_jpy"], "s5_net": s5["net_jpy"]})

    result = {"contract": "QR_PAIR_ADAPTATION_LAB_V1",
              "grid_size_per_pair": len(GRID),
              "phase1_screened": len(NEW_PAIRS),
              "phase2_candidates": candidates,
              "selected": {k: v for k, v in selected.items()},
              "all_gates_survivors": survivors,
              "board": board}
    (out_root / "pair_adaptation_board.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"SURVIVORS: {json.dumps(survivors, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
