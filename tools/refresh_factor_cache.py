#!/usr/bin/env python3
"""
Fetch OANDA candles and compute technicals for Claude discretionary trading.

Usage:
    python3 tools/refresh_factor_cache.py [PAIR ...] [--quiet]
    python3 tools/refresh_factor_cache.py --all --quiet

- Single pair: updates factor_cache in-memory (USD_JPY) + writes per-pair JSON
- Multiple pairs / --all: writes per-pair JSON for each, factor_cache for last pair
- Per-pair JSON: logs/technicals_{PAIR}.json (RSI, ADX, ATR, EMA slopes, regime etc.)
"""
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import tomllib as tomli  # Python 3.11+
except ImportError:
    try:
        import tomli  # backport
    except ImportError:
        tomli = None  # fallback to manual parsing
try:
    import pandas as pd
except ModuleNotFoundError:
    venv_python = ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if venv_python.exists() and current_python != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise
try:
    from indicators.factor_cache import on_candle, all_factors
except ImportError:
    on_candle = None  # analysis module archived — factor_cache disabled
    all_factors = None
from indicators.calc_core import IndicatorEngine

OANDA_BASE = "https://api-fxtrade.oanda.com"
ALL_PAIRS = [
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
    "NZD_JPY", "CAD_JPY",
    "EUR_CHF", "AUD_NZD", "AUD_CAD",
]

EXPORT_KEYS = [
    "rsi", "atr", "atr_pips", "adx", "plus_di", "minus_di", "bbw",
    "ema_slope_5", "ema_slope_10", "ema_slope_20", "macd", "macd_hist",
    "stoch_rsi", "cci", "roc5", "roc10", "regime", "close",
    "bb_upper", "bb_lower", "bb_mid", "bb_span_pips", "vwap_gap",
    "ichimoku_cloud_pos", "ichimoku_span_a_gap", "ichimoku_span_b_gap",
    "div_rsi_score", "div_rsi_kind", "div_macd_score", "div_macd_kind",
    "div_score",
    # 新指標（武器庫拡張）
    "kc_width", "chaikin_vol", "donchian_width",
    "cluster_high_gap", "cluster_low_gap",
    "upper_wick_avg_pips", "lower_wick_avg_pips",
    "high_hits", "low_hits", "high_hit_interval", "low_hit_interval",
    "swing_dist_high", "swing_dist_low",
]


def _load_config():
    if tomli is not None:
        with open(ROOT / "config" / "env.toml", "rb") as f:
            cfg = tomli.load(f)
    else:
        # Fallback: manual TOML parsing (Python 3.10 without tomli)
        cfg = {}
        with open(ROOT / "config" / "env.toml") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg["oanda_token"], cfg["oanda_account_id"]


def _fetch_candles(token: str, instrument: str, granularity: str, count: int) -> list:
    url = (
        f"{OANDA_BASE}/v3/instruments/{instrument}/candles"
        f"?granularity={granularity}&count={count}&price=M"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", False) and granularity != "M1":
                continue
            mid = c["mid"]
            candles.append({
                "time": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
            })
        return candles
    except Exception as e:
        print(f"ERROR fetching {instrument} {granularity}: {e}", file=sys.stderr)
        return []


TF_MAP = {
    "M1": 200,
    "M5": 200,
    "M15": 200,
    "H1": 100,
    "H4": 50,
}


def _compute_technicals(candles: list) -> dict:
    """Compute indicators from a list of candle dicts using IndicatorEngine."""
    if len(candles) < 20:
        return {}
    df = pd.DataFrame(candles)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    try:
        factors = IndicatorEngine.compute(df)
        result = {}
        for k in EXPORT_KEYS:
            v = factors.get(k)
            if v is not None:
                if isinstance(v, float):
                    result[k] = round(v, 5)
                else:
                    result[k] = v
        if candles:
            result["close"] = candles[-1]["close"]
            result["timestamp"] = candles[-1].get("time", "")
        return result
    except Exception as e:
        print(f"ERROR computing technicals: {e}", file=sys.stderr)
        return {}


def _save_pair_technicals(pair: str, data: dict):
    """Save per-pair technicals to logs/technicals_{PAIR}.json"""
    path = ROOT / "logs" / f"technicals_{pair}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _refresh_pair_sync(token: str, pair: str, update_factor_cache: bool = False, quiet: bool = False):
    """Fetch candles for one pair, compute technicals, save JSON. (sync, thread-safe)"""
    pair_data = {"pair": pair, "timeframes": {}}
    total = 0

    for tf, count in TF_MAP.items():
        candles = _fetch_candles(token, pair, tf, count)
        if not candles:
            continue

        technicals = _compute_technicals(candles)
        if technicals:
            pair_data["timeframes"][tf] = technicals

        total += len(candles)

    if pair_data["timeframes"]:
        _save_pair_technicals(pair, pair_data)
        if not quiet:
            m1 = pair_data["timeframes"].get("M1", {})
            print(f"  {pair}: M1 close={m1.get('close')} rsi={m1.get('rsi', 'N/A')}")

    return total


async def refresh_pair(token: str, pair: str, update_factor_cache: bool = False, quiet: bool = False):
    """Async wrapper — runs sync fetch in thread pool for true parallelism."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _refresh_pair_sync, token, pair, update_factor_cache, quiet
    )


async def refresh(pairs: list[str], quiet: bool = False):
    token, _ = _load_config()

    # Run all pairs concurrently (I/O-bound OANDA API calls)
    tasks = []
    for pair in pairs:
        update_fc = (pair == "USD_JPY")
        tasks.append(refresh_pair(token, pair, update_factor_cache=update_fc, quiet=quiet))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total = 0
    for r in results:
        if isinstance(r, Exception):
            if not quiet:
                print(f"  ERROR: {r}", file=sys.stderr)
        else:
            total += r

    if not quiet:
        print(f"\nDone. {total} candles across {len(pairs)} pairs.")
    return total


def main():
    pairs = []
    quiet = False
    use_all = False
    tf_filter = None  # None = all timeframes
    for arg in sys.argv[1:]:
        if arg == "--quiet":
            quiet = True
        elif arg == "--all":
            use_all = True
        elif arg.startswith("--tf="):
            tf_filter = arg[5:].split(",")  # e.g. --tf=M1,M5
        elif not arg.startswith("-"):
            pairs.append(arg)

    if use_all:
        pairs = ALL_PAIRS
    elif not pairs:
        pairs = ["USD_JPY"]

    if tf_filter:
        global TF_MAP
        TF_MAP = {k: v for k, v in TF_MAP.items() if k in tf_filter}

    if not quiet:
        print(f"Refreshing technicals for {', '.join(pairs)} [{','.join(TF_MAP.keys())}]...")
    asyncio.run(refresh(pairs, quiet))
    # Always print a completion line so scheduled tasks don't stall on empty output
    print(f"OK {len(pairs)}pairs {','.join(TF_MAP.keys())}")


if __name__ == "__main__":
    main()
