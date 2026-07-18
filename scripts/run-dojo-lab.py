#!/usr/bin/env python3
"""DOJO lab: run the declared experiment grid until something wins honestly.

Discipline: iterate freely on TRAIN (that is what a dojo is for), then
confirm ONLY the TRAIN-positive configs on the untouched VAL window.
The scoreboard records every config tried (multiplicity is visible, not
hidden).  A config "wins" only if it is net-positive on TRAIN and VAL.

Declared grid (12 configs, fixed before any run):
  signals  burst | pullback_limit | range_fade_limit
  exits    E1 TP3/no-SL/ceiling240m   (operator's SL-free philosophy)
           E2 TP6/no-SL/ceiling480m
           E3 TP4/SL12/ceiling480m    (wide-SL 1:3)
           E4 TP10/SL30/ceiling1440m  (very wide)
  (range_fade uses E1-E4 with both-side limits; 3 x 4 = 12)

TRAIN 2024-01-02 .. 2025-07-01, VAL 2025-07-01 .. 2026-07-01 on USD_JPY.
Each run is a full DOJO session (real M1 quotes, fills at touched quotes
only, hash-chained ledger, OHLC intrabar; both-touch ambiguity is small
for these geometries and bracketed on winners with OLHC).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
TRAIN = ("2024-01-02T00:00:00", "2025-07-01T00:00:00")
VAL = ("2025-07-01T00:00:00", "2026-07-01T00:00:00")

EXITS = {
    "E1_tp3_nosl_c240": {"tp_pips": 3, "sl_pips": None, "ceiling_min": 240},
    "E2_tp6_nosl_c480": {"tp_pips": 6, "sl_pips": None, "ceiling_min": 480},
    "E3_tp4_sl12_c480": {"tp_pips": 4, "sl_pips": 12, "ceiling_min": 480},
    "E4_tp10_sl30_c1440": {"tp_pips": 10, "sl_pips": 30, "ceiling_min": 1440},
}
SIGNALS = ["burst", "pullback_limit", "range_fade_limit"]


def run_config(name: str, cfg: dict, window: tuple[str, str], out_root: Path,
               intrabar: str = "OHLC") -> dict:
    session = out_root / f"{name}_{window[0][:10]}_{intrabar}"
    session.mkdir(parents=True, exist_ok=True)
    (session / "inbox").mkdir(exist_ok=True)
    env = dict(os.environ)
    env["DOJO_BOT_CONFIG"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        [PY, str(REPO / "scripts/run-virtual-market-session.py"),
         "--feed", "replay", "--session-dir", str(session),
         "--pairs", "USD_JPY",
         "--from", window[0], "--to", window[1],
         "--bars-per-second", "100000", "--state-every", "5000",
         "--fast-ledger", "--intrabar", intrabar,
         "--bot-module", str(REPO / "bots/lab_bot.py") + ":Bot"],
        capture_output=True, text=True, env=env, timeout=7200,
    )
    if proc.returncode != 0:
        return {"name": name, "error": proc.stderr[-300:]}
    pl = wins = trades = closeouts = 0
    final_balance = None
    for line in (session / "ledger.jsonl").open():
        rec = json.loads(line)
        payload = rec["payload"] if isinstance(rec["payload"], dict) else {}
        p = payload.get("pl_jpy")
        if p is not None and rec["event"].startswith(("EXIT", "CLOSE", "MARGIN")):
            pl += p; trades += 1; wins += int(p > 0)
        if rec["event"] == "MARGIN_CLOSEOUT":
            closeouts += 1
        if rec["event"] == "SESSION_STOP" and payload.get("account"):
            final_balance = payload["account"].get("balance_jpy")
    return {
        "name": name, "window": window[0][:10] + ".." + window[1][:10],
        "intrabar": intrabar,
        "trades": trades, "wins": wins,
        "win_rate": round(wins / trades, 4) if trades else None,
        "net_jpy": round(pl), "final_balance_jpy": final_balance,
        "margin_closeouts": closeouts,
    }


def main() -> int:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    scoreboard: list[dict] = []
    grid = []
    for sig in SIGNALS:
        for exit_name, exit_cfg in EXITS.items():
            cfg = {"signal": sig, **exit_cfg, "max_concurrent": 3,
                   "per_pos_lev": 4.3, "atr_floor_pips": 1.0,
                   "pull_atr": 0.6, "fade_atr": 1.2, "eff_max": 0.2}
            grid.append((f"{sig}__{exit_name}", cfg))
    print(f"declared grid: {len(grid)} configs", flush=True)

    for name, cfg in grid:
        row = run_config(name, cfg, TRAIN, out_root)
        scoreboard.append({"phase": "TRAIN", **row, "config": cfg})
        print(json.dumps(scoreboard[-1], ensure_ascii=False), flush=True)

    train_positive = [
        s for s in scoreboard
        if s["phase"] == "TRAIN" and (s.get("net_jpy") or 0) > 0
    ]
    print(f"TRAIN-positive: {len(train_positive)}/{len(grid)}", flush=True)
    for s in train_positive:
        row = run_config(s["name"], s["config"], VAL, out_root)
        scoreboard.append({"phase": "VAL", **row, "config": s["config"]})
        print(json.dumps(scoreboard[-1], ensure_ascii=False), flush=True)
        # ambiguity bracket on VAL survivors
        if (row.get("net_jpy") or 0) > 0:
            row2 = run_config(s["name"], s["config"], VAL, out_root, intrabar="OLHC")
            scoreboard.append({"phase": "VAL_OLHC", **row2, "config": s["config"]})
            print(json.dumps(scoreboard[-1], ensure_ascii=False), flush=True)

    out = out_root / "scoreboard.json"
    out.write_text(json.dumps({
        "contract": "QR_DOJO_LAB_V1",
        "declared_grid_size": len(grid),
        "train_window": TRAIN, "val_window": VAL,
        "multiplicity_note": "every config tried is listed; winners must be "
                             "net-positive on TRAIN and untouched VAL (and "
                             "survive the OLHC ambiguity bracket)",
        "scoreboard": scoreboard,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"scoreboard -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
