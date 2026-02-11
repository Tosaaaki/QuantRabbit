#!/usr/bin/env python3
"""
Train a multi-horizon USD/JPY forecast bundle (scikit-learn).

Input data
----------
Prefers replay candle logs produced by market_data.replay_logger:
  logs/replay/USD_JPY/USD_JPY_M5_YYYYMMDD.jsonl
  logs/replay/USD_JPY/USD_JPY_H1_YYYYMMDD.jsonl
  logs/replay/USD_JPY/USD_JPY_D1_YYYYMMDD.jsonl
If the replay logs are missing/insufficient, backfill them from OANDA:
  python scripts/backfill_replay_candles.py --instrument USD_JPY --timeframes M5,H1,D1

Outputs
-------
Joblib bundle containing 5 horizon models.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from analysis.forecast_sklearn import DEFAULT_HORIZONS, save_bundle, train_bundle


ROOT = Path(__file__).resolve().parents[1]


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _parse_day_from_name(path: Path) -> dt.date | None:
    # Expected: USD_JPY_M5_YYYYMMDD.jsonl
    stem = path.stem
    parts = stem.split("_")
    if not parts:
        return None
    day = parts[-1]
    if len(day) != 8 or not day.isdigit():
        return None
    try:
        return dt.date(int(day[0:4]), int(day[4:6]), int(day[6:8]))
    except Exception:
        return None


def load_replay_candles(
    instrument: str,
    timeframe: str,
    *,
    replay_dir: Path,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    base = replay_dir / instrument
    if not base.exists():
        return []

    tf = timeframe.strip().upper()
    pattern = f"{instrument}_{tf}_*.jsonl"
    files = sorted(base.glob(pattern))
    if not files:
        return []

    cutoff = _utc_today() - dt.timedelta(days=max(1, int(lookback_days)))
    wanted: list[Path] = []
    for fp in files:
        day = _parse_day_from_name(fp)
        if day is None or day >= cutoff:
            wanted.append(fp)

    out: list[Dict[str, Any]] = []
    for fp in wanted:
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("timestamp") or row.get("time") or row.get("ts")
                if ts is None:
                    continue
                out.append(
                    {
                        "timestamp": ts,
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                    }
                )
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="USD_JPY")
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "logs" / "replay")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "config" / "forecast_models" / "USD_JPY_bundle.joblib",
    )
    ap.add_argument("--lookback-m5-days", type=int, default=120)
    ap.add_argument("--lookback-h1-days", type=int, default=540)
    ap.add_argument("--lookback-d1-days", type=int, default=2000)
    args = ap.parse_args()

    candles_by_tf: Mapping[str, List[Dict[str, Any]]] = {
        "M5": load_replay_candles(
            args.instrument,
            "M5",
            replay_dir=args.replay_dir,
            lookback_days=args.lookback_m5_days,
        ),
        "H1": load_replay_candles(
            args.instrument,
            "H1",
            replay_dir=args.replay_dir,
            lookback_days=args.lookback_h1_days,
        ),
        "D1": load_replay_candles(
            args.instrument,
            "D1",
            replay_dir=args.replay_dir,
            lookback_days=args.lookback_d1_days,
        ),
    }

    missing = [tf for tf, rows in candles_by_tf.items() if not rows]
    if missing:
        raise SystemExit(
            "missing_replay_candles: "
            + ", ".join(missing)
            + f" (expected under {args.replay_dir}/{args.instrument}/)"
        )

    bundle, reports = train_bundle(
        args.instrument,
        candles_by_tf,
        horizons=DEFAULT_HORIZONS,
    )
    save_bundle(bundle, args.out)

    print("=== Forecast bundle trained ===")
    print(f"instrument={bundle.instrument} created_at={bundle.created_at} out={args.out}")
    for spec in DEFAULT_HORIZONS:
        r = reports.get(spec.name, {})
        logloss = r.get("logloss")
        brier = r.get("brier")
        auc = r.get("auc")
        n = int(r.get("n", 0))
        pos_rate = r.get("pos_rate_test")
        line = (
            f"- {spec.name:>2} tf={spec.timeframe:<2} step={spec.step_bars:<4} "
            f"n={n:<6} logloss={logloss:.4f} brier={brier:.4f} "
        )
        if auc is not None:
            line += f"auc={auc:.4f} "
        if pos_rate is not None:
            line += f"pos_rate_test={pos_rate:.3f}"
        print(line.strip())


if __name__ == "__main__":
    main()
