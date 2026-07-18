#!/usr/bin/env python3
"""Faithful 6.5-year replay of the golden-day (2025-12-09) MomentumBurst.

Fidelity contract:
  * The DECIDER is the vendored, verbatim strategy class from legacy
    commit 41644097 (repo HEAD five hours before the 53/53 run) —
    ``MomentumBurstMicro.check`` is imported and called unmodified.
  * Factors are computed with the vendored IndicatorEngine formulas
    (_rsi/_atr/_adx/ma/ema/vol_5m copied by import, full-series
    vectorized; the only divergence from the live 2000-bar deque is ewm
    tail truncation, negligible at 2000 bars and declared here).
  * A vectorized pre-gate (necessary conditions of the rule) selects
    candidate bars; the vendored check() is the sole authority on them.
  * Execution: signal on M1 close -> enter next M1 open (ask for long,
    bid for short); broker TP/SL from the signal's own tp/sl pips;
    pessimistic resolution (SL first whenever both touch a bar; TP no
    earlier than the bar after entry; SL allowed on the entry bar).
  * Declared approximations: M1-bar cadence instead of the live 30s tick
    loop; strategy-health confidence scaling absent (scale 1.0); drift
    and mtf keys absent exactly as in the live fac_m1 of that commit.

Arms:
  AS_LIVED   the strategy's own TP/SL only; unlimited stacking; margin
             closeout at 100% usage is the only other exit.
  PROTECTED  adds the arsenal cage: max 3 concurrent, 4h hard ceiling
             market close, 92% usage cap refuses new entries.

Money: 200k JPY NAV, NAV-proportional 4.3x per position, 25x margin.
Splits: TRAIN 2020-2023 / VAL 2024-2025 / TEST_2026H1 (single declared
config, no search).  Includes a golden-day fidelity report comparing
2025-12-09 replay trades against the live 53/53 record.  Shadow only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

UTC = timezone.utc
PAIR = "USD_JPY"
PIP = 0.01
NAV_START = 200_000.0
PER_POSITION_LEVERAGE = 4.3
LEVERAGE = 25.0
USAGE_CAP = 0.92
HARD_CEILING_S = 4 * 3600
MAX_CONCURRENT_PROTECTED = 3
REPLAY_FROM = "2020-03-02"
SPLITS = {
    "TRAIN_2020_2023": ("2020-03-02", "2024-01-01"),
    "VAL_2024_2025": ("2024-01-01", "2026-01-01"),
    "TEST_2026H1": ("2026-01-01", "2026-07-10"),
}
GOLDEN_DAY = "2025-12-09"
VENDOR = Path(__file__).resolve().parents[1] / "vendored" / "golden_20251209"


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_vendored():
    """Import the verbatim strategy with an 'analysis' shim package."""

    analysis_pkg = ModuleType("analysis")
    analysis_pkg.__path__ = []  # mark as package
    sys.modules["analysis"] = analysis_pkg
    ma_projection = _load_module(
        "analysis.ma_projection", VENDOR / "analysis_ma_projection.py"
    )
    analysis_pkg.ma_projection = ma_projection
    calc_core = _load_module("golden_calc_core", VENDOR / "indicators_calc_core.py")
    strategy = _load_module(
        "golden_momentum_burst", VENDOR / "strategies_micro_momentum_burst.py"
    )
    return calc_core, strategy


def _load_rows(root: Path):
    rows = []
    for shard_file in sorted(root.glob(f"*/{PAIR}/{PAIR}_M1_BA_*.jsonl.gz")):
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
    # de-duplicate epochs (shard boundaries)
    dedup = []
    last_epoch = None
    for r in rows:
        if r[0] != last_epoch:
            dedup.append(r)
            last_epoch = r[0]
    return dedup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be clean; refusing stale reuse")

    calc_core, strategy_mod = _import_vendored()
    Strategy = strategy_mod.MomentumBurstMicro

    rows = _load_rows(args.root)
    n = len(rows)
    epochs = np.array([r[0] for r in rows], dtype=np.int64)
    bid_o = np.array([r[1] for r in rows]); bid_h = np.array([r[2] for r in rows])
    bid_l = np.array([r[3] for r in rows]); bid_c = np.array([r[4] for r in rows])
    ask_o = np.array([r[5] for r in rows]); ask_h = np.array([r[6] for r in rows])
    ask_l = np.array([r[7] for r in rows]); ask_c = np.array([r[8] for r in rows])
    mid_o = (bid_o + ask_o) / 2.0; mid_h = (bid_h + ask_h) / 2.0
    mid_l = (bid_l + ask_l) / 2.0; mid_c = (bid_c + ask_c) / 2.0

    close = pd.Series(mid_c); high = pd.Series(mid_h); low = pd.Series(mid_l)
    ma10 = close.rolling(10, min_periods=10).mean().to_numpy()
    ma20 = close.rolling(20, min_periods=20).mean().to_numpy()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    vol_5m = (close.diff().abs().rolling(5, min_periods=5).mean() / 0.01).to_numpy()
    rsi = calc_core._rsi(close, 14).to_numpy()
    atr = calc_core._atr(high, low, close, 14).to_numpy()
    adx = calc_core._adx(high, low, close, 14).to_numpy()
    spread_pips = (ask_c - bid_c) / PIP

    # Vectorized necessary-condition pre-gate (supersets the rule).
    atr_pips = atr * 100.0
    gap_pips = (ma10 - ma20) / PIP
    # No spread term here: the pre-gate must superset BOTH participation
    # modes; check() is the sole spread authority (gate-on mode).
    base = (
        (atr_pips >= 1.2) & (vol_5m >= 0.7) & (adx >= 24.0)
        & ~np.isnan(ma10) & ~np.isnan(ma20)
    )
    long_gate = base & (gap_pips >= 0.32) & (mid_c > ema20 + 0.0015) & (rsi >= 54)
    short_gate = base & (gap_pips <= -0.32) & (mid_c < ema20 - 0.0015) & (rsi <= 46)
    candidates = np.nonzero(long_gate | short_gate)[0]

    def build_signals(gate_off: bool) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for k in candidates:
            if k < 25 or k + 1 >= n:
                continue
            fac = {
                "close": float(mid_c[k]), "open": float(mid_o[k]),
                "high": float(mid_h[k]), "low": float(mid_l[k]),
                "ma10": float(ma10[k]), "ma20": float(ma20[k]),
                "ema20": float(ema20[k]), "rsi": float(rsi[k]),
                "atr": float(atr[k]), "adx": float(adx[k]),
                "vol_5m": float(vol_5m[k]),
                "spread_pips": 0.0 if gate_off else float(spread_pips[k]),
                "candles": [
                    {"high": float(mid_h[j]), "low": float(mid_l[j]),
                     "open": float(mid_o[j]), "close": float(mid_c[j])}
                    for j in range(k - 3, k + 1)
                ],
            }
            signal = Strategy.check(fac)
            if signal and signal.get("action") in {"OPEN_LONG", "OPEN_SHORT"}:
                out[int(k)] = signal
        return out

    # Fidelity finding: with the real corpus spread in fac the golden day
    # yields 1 signal vs the live 53; with spread_pips=0.0 (the live
    # worker's spread monitor supplied nothing, so its gate was
    # effectively OFF) it yields 58 =~ live. GATE_OFF_AS_LIVED is the
    # fidelity-validated participation mode; execution still pays the
    # full real bid/ask cost in both modes.
    signals_gate_on = build_signals(gate_off=False)
    signals = build_signals(gate_off=True)  # arms below use as-lived mode

    def run_arm(protected: bool) -> dict[str, Any]:
        balance = NAV_START
        open_pos: list[dict[str, Any]] = []
        daily: dict[str, float] = {}
        trades = wins = 0
        closeouts = 0
        dead_day = None
        golden_trades: list[dict[str, Any]] = []

        def day_of(i: int) -> str:
            return datetime.fromtimestamp(int(epochs[i]), tz=UTC).date().isoformat()

        def close_position(pos, price, i, reason):
            nonlocal balance, trades, wins
            side = pos["side"]
            pl = (price - pos["entry"]) * pos["units"] if side == "LONG" else (
                pos["entry"] - price) * pos["units"]
            balance += pl
            day = day_of(i)
            daily[day] = daily.get(day, 0.0) + pl
            trades += 1
            wins += int(pl > 0)
            if day == GOLDEN_DAY:
                golden_trades.append({"pl_jpy": round(pl), "reason": reason})

        replay_from_epoch = int(datetime.fromisoformat(
            REPLAY_FROM + "T00:00:00+00:00").timestamp())
        for i in range(n):
            if epochs[i] < replay_from_epoch:
                continue
            if balance <= NAV_START * 0.02:
                dead_day = dead_day or day_of(i)
                break
            # resolve exits (positions entered before this bar)
            still = []
            for pos in open_pos:
                if pos["bar"] >= i:
                    still.append(pos)
                    continue
                side = pos["side"]
                sl_hit = (bid_l[i] <= pos["sl"]) if side == "LONG" else (ask_h[i] >= pos["sl"])
                tp_hit = (bid_h[i] >= pos["tp"]) if side == "LONG" else (ask_l[i] <= pos["tp"])
                if sl_hit:  # pessimistic SL-first
                    close_position(pos, pos["sl"], i, "SL")
                elif tp_hit and i > pos["bar"] + 1:
                    close_position(pos, pos["tp"], i, "TP")
                elif protected and epochs[i] - pos["entry_epoch"] >= HARD_CEILING_S:
                    px = bid_o[i] if side == "LONG" else ask_o[i]
                    close_position(pos, px, i, "HARD_CEILING")
                else:
                    still.append(pos)
            open_pos = still

            # margin state
            mid = mid_c[i]
            unreal = sum(
                (mid - p["entry"]) * p["units"] if p["side"] == "LONG"
                else (p["entry"] - mid) * p["units"] for p in open_pos
            )
            equity = balance + unreal
            margin = sum(p["units"] for p in open_pos) * mid / LEVERAGE
            if equity <= 0 or (margin > 0 and margin / max(equity, 1e-9) >= 1.0):
                for p in open_pos:
                    px = bid_c[i] if p["side"] == "LONG" else ask_c[i]
                    close_position(p, px, i, "MARGIN_CLOSEOUT")
                open_pos = []
                closeouts += 1
                continue

            # entries: signal fired on the PREVIOUS bar's close
            sig = signals.get(i - 1)
            if not sig:
                continue
            if protected and len(open_pos) >= MAX_CONCURRENT_PROTECTED:
                continue
            if margin / max(equity, 1e-9) >= USAGE_CAP:
                continue
            side = "LONG" if sig["action"] == "OPEN_LONG" else "SHORT"
            entry = float(ask_o[i]) if side == "LONG" else float(bid_o[i])
            sl_pips = float(sig["sl_pips"]); tp_pips = float(sig["tp_pips"])
            units = max(equity, 0.0) * PER_POSITION_LEVERAGE / entry
            open_pos.append({
                "side": side, "entry": entry, "bar": i, "entry_epoch": int(epochs[i]),
                "units": units,
                "sl": entry - sl_pips * PIP if side == "LONG" else entry + sl_pips * PIP,
                "tp": entry + tp_pips * PIP if side == "LONG" else entry - tp_pips * PIP,
            })

        def split(from_s: str, to_s: str) -> dict[str, Any]:
            days = {d: v for d, v in daily.items() if from_s <= d < to_s}
            equity_track = NAV_START
            compound = 1.0
            worst = best = 0.0
            negd = 0
            for d in sorted(days):
                r = max(days[d] / max(equity_track, 1e-9), -1.0)
                equity_track = max(equity_track + days[d], 0.0)
                compound = max(compound * (1.0 + r), 0.0)
                worst = min(worst, r); best = max(best, r)
                negd += int(days[d] < 0)
            months = max(1e-9, len(days) / 21.7)
            return {
                "net_jpy": round(sum(days.values())),
                "nav_multiple": round(compound, 4),
                "monthly_multiple": round(compound ** (1.0 / months), 4) if days and compound > 0 else 0.0,
                "active_days": len(days), "negative_days": negd,
                "worst_day_nav": round(worst, 4), "best_day_nav": round(best, 4),
            }

        golden_wins = sum(1 for t in golden_trades if t["pl_jpy"] > 0)
        return {
            "trades": trades,
            "win_rate": round(wins / trades, 4) if trades else None,
            "margin_closeouts": closeouts,
            "dead_day": dead_day,
            "final_balance_jpy": round(balance),
            "golden_day_fidelity": {
                "replay_trades": len(golden_trades),
                "replay_wins": golden_wins,
                "live_record": "53 trades, 53 wins, +24,542 JPY",
                "replay_net_jpy": round(sum(t["pl_jpy"] for t in golden_trades)),
            },
            "splits": {name: split(f, t) for name, (f, t) in SPLITS.items()},
        }

    as_lived = run_arm(protected=False)
    protected = run_arm(protected=True)
    signals_backup = signals
    signals = signals_gate_on
    gate_on_protected = run_arm(protected=True)
    signals = signals_backup

    body: dict[str, Any] = {
        "contract": "QR_GOLDEN_MOMENTUM_BURST_REPLAY_V1",
        "schema_version": 1,
        "pair": PAIR,
        "decider": "vendored verbatim MomentumBurstMicro.check from legacy commit 41644097",
        "signal_bars_total_gate_off_as_lived": len(signals),
        "signal_bars_total_gate_on": len(signals_gate_on),
        "participation_mode_note": (
            "GATE_OFF_AS_LIVED validated by golden-day fidelity (58 vs live "
            "53 trades); execution pays real bid/ask cost in all arms"
        ),
        "declared_approximations": [
            "M1-bar cadence instead of live 30s tick loop",
            "full-series ewm vs live 2000-bar deque (negligible tail)",
            "strategy-health confidence scale fixed at 1.0",
            "drift/mtf keys absent exactly as live fac_m1 at that commit",
            "TP no earlier than the bar after entry (pessimistic)",
        ],
        "accounting": {
            "nav_start_jpy": NAV_START, "per_position_leverage": PER_POSITION_LEVERAGE,
            "leverage": LEVERAGE, "usage_cap": USAGE_CAP, "sizing": "NAV_PROPORTIONAL",
        },
        "as_lived": as_lived,
        "protected": protected,
        "gate_on_protected": gate_on_protected,
        "single_declared_config_no_search": True,
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
    print(json.dumps({
        "status": "GOLDEN_REPLAY_SEALED",
        "signals": len(signals),
        "as_lived": {k: as_lived[k] for k in ("trades", "win_rate", "margin_closeouts", "dead_day", "golden_day_fidelity")},
        "protected": {k: protected[k] for k in ("trades", "win_rate", "margin_closeouts", "dead_day", "golden_day_fidelity")},
        "as_lived_splits": as_lived["splits"],
        "protected_splits": protected["splits"],
        "gate_on_protected_splits": gate_on_protected["splits"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
