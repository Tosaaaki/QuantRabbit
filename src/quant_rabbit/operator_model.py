"""Feature computation for the frozen operator entry model.

One copy, shared by training and by the live signal path. A silent mismatch
between how a feature is computed at fit time and at run time is the failure
mode that produces a model which back-tests and does not trade, so the two
paths call the same function here rather than each holding their own version.

Everything is point-in-time: a bar is used only if it closed strictly before
the decision timestamp, and every distance is expressed in ATR units so the
values are comparable across pairs and volatility regimes.

Read-only. This module fetches candles and returns numbers; it never places an
order and never imports an execution client.
"""

from __future__ import annotations

import json
import re
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ENV = Path("/Users/tossaki/App/QuantRabbit/.env.local")

# Horizons in hours for the generic price block.
RETURN_WINDOWS = (4, 12, 24, 72, 168)
# The frozen exit: 336 hours. Shorter fixed horizons measured negative
# (8h -2.38, 24h -2.17, 72h -2.10 pips); 336h gives +28.25.
EXIT_HOURS = 336
MODEL_PATH = Path(__file__).resolve().parents[2] / "research" / "operator_model" / "model_v1.pkl"


def _creds():
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip()
    return env["QR_OANDA_TOKEN"], env.get("QR_OANDA_BASE_URL", "https://api-fxtrade.oanda.com")


def _get(path: str, query: dict) -> dict:
    token, base = _creds()
    url = f"{base}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(re.sub(r"\.(\d{6})\d*", r".\1", s).replace("Z", "+00:00"))


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def closed_by(candle: dict, at: datetime, granularity: timedelta) -> bool:
    """True only if this bar had finished before `at`.

    OANDA stamps a candle with its OPEN time, so the bar labelled 09:00 covers
    09:00-10:00. Filtering on `time < at` therefore admits the bar straddling the
    decision — at 09:30 it hands back the 10:00 close, thirty minutes of future.
    Historical bars come back `complete: true` regardless of the `to` parameter,
    so completeness is not the guard; the bar's END has to be compared.

    This was live for the whole first build. It leaked a mean of 9.09 pips into
    `px` (max 183.80) across 98.4% of rows, and through `px` into all 52
    features and the entry price of the label. It did NOT affect the live path,
    where the forming bar is genuinely incomplete — so it was a train/live
    mismatch that made the backtest optimistic, which is the exact failure the
    shared-feature design in this module was written to prevent.
    """
    if not candle.get("complete"):
        return False
    return parse_time(candle["time"]) + granularity <= at


def _atr(highs, lows, closes, n) -> float:
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
          for i in range(1, len(closes))]
    return sum(tr[-n:]) / n if len(tr) >= n else 0.0


def price_block(pair: str, at: datetime) -> dict | None:
    """Generic H1 price shape: returns, range position, breakout, volatility."""
    payload = _get(f"/v3/instruments/{pair}/candles",
                   {"granularity": "H1", "to": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "count": "400", "price": "M"})
    bars = [c for c in payload.get("candles", []) if closed_by(c, at, timedelta(hours=1))]
    if len(bars) < 200:
        return None
    h = [float(c["mid"]["h"]) for c in bars]
    l = [float(c["mid"]["l"]) for c in bars]
    c = [float(c["mid"]["c"]) for c in bars]
    atr = _atr(h, l, c, 14)
    if atr <= 0:
        return None
    px = c[-1]
    f: dict[str, float] = {}
    for n in RETURN_WINDOWS:
        f[f"ret{n}"] = (c[-1] - c[-1 - n]) / atr
        f[f"pos{n}"] = (px - min(l[-n:])) / max(max(h[-n:]) - min(l[-n:]), 1e-9)
        f[f"brk{n}"] = (px - max(h[-n:-1])) / atr
        f[f"vol{n}"] = statistics.pstdev(c[-n:]) / atr
    f["ema_gap"] = (sum(c[-12:]) / 12 - sum(c[-48:]) / 48) / atr
    f["atr_ratio"] = atr / max(_atr(h, l, c, 168), 1e-12)
    f["hour"] = float(at.hour)
    f["dow"] = float(at.weekday())
    f["dist_hi168"] = (px - max(h[-168:])) / atr
    f["dist_lo168"] = (px - min(l[-168:])) / atr
    return f


def level_block(pair: str, at: datetime) -> dict | None:
    """Distance to every level a discretionary trader might lean on.

    Adding this block moved the model's AUC from 0.584 to 0.689, and the
    nearest swing low, session VWAP and previous-day low rank among the
    strongest features - the family the operator named when asked what he
    watches. Which specific level matters was never settled by asking, so all
    of them are recorded and the fit decides.
    """
    payload = _get(f"/v3/instruments/{pair}/candles",
                   {"granularity": "M15", "to": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "count": "600", "price": "M"})
    bars = [c for c in payload.get("candles", []) if closed_by(c, at, timedelta(minutes=15))]
    if len(bars) < 300:
        return None
    pip = pip_size(pair)
    h = [float(c["mid"]["h"]) for c in bars]
    l = [float(c["mid"]["l"]) for c in bars]
    c_ = [float(c["mid"]["c"]) for c in bars]
    vol = [int(c.get("volume", 0)) for c in bars]
    atr = _atr(h, l, c_, 56)
    if atr <= 0:
        return None
    px = c_[-1]
    day = at.strftime("%Y-%m-%d")
    prev = (at - timedelta(days=1)).strftime("%Y-%m-%d")

    def band(pred):
        idx = [i for i, c in enumerate(bars) if pred(parse_time(c["time"]))]
        return (max(h[i] for i in idx), min(l[i] for i in idx), c_[idx[0]]) if idx else (None, None, None)

    pdh, pdl, _x = band(lambda t: t.strftime("%Y-%m-%d") == prev)
    tdh, tdl, tdo = band(lambda t: t.strftime("%Y-%m-%d") == day)
    hour = at.hour
    sess_start = {0: 0, 1: 0, 2: 7, 3: 12, 4: 16}[min(4, max(0, hour // 4))] if hour >= 7 else 0
    ssh, ssl, sso = band(lambda t: t.strftime("%Y-%m-%d") == day and t.hour >= sess_start)

    sw_h = [h[i] for i in range(len(h) - 200, len(h) - 3) if h[i] == max(h[i - 3:i + 4])]
    sw_l = [l[i] for i in range(len(l) - 200, len(l) - 3) if l[i] == min(l[i - 3:i + 4])]
    near_h = min((v for v in sw_h if v >= px), default=None)
    near_l = max((v for v in sw_l if v <= px), default=None)

    g50, g100 = 50 * pip, 100 * pip
    sess_idx = [i for i, c in enumerate(bars)
                if parse_time(c["time"]).strftime("%Y-%m-%d") == day
                and parse_time(c["time"]).hour >= sess_start]
    tv = sum(vol[i] for i in sess_idx)
    vwap = (sum(((h[i] + l[i] + c_[i]) / 3) * vol[i] for i in sess_idx) / tv) if tv else None

    f: dict[str, float] = {}
    for name, level in (("pdh", pdh), ("pdl", pdl), ("tdh", tdh), ("tdl", tdl), ("tdo", tdo),
                        ("ssh", ssh), ("ssl", ssl), ("sso", sso), ("swh", near_h), ("swl", near_l),
                        ("r50", round(px / g50) * g50), ("r100", round(px / g100) * g100),
                        ("vwap", vwap)):
        f[f"d_{name}"] = (px - level) / atr if level is not None else 0.0
        f[f"has_{name}"] = 1.0 if level is not None else 0.0
    return f


def features(pair: str, at: datetime | None = None) -> dict | None:
    """The full 52-feature vector, or None if history is too thin."""
    at = at or datetime.now(timezone.utc)
    a = price_block(pair, at)
    if a is None:
        return None
    b = level_block(pair, at)
    if b is None:
        return None
    return {**a, **b}


def load_model(path: Path | None = None):
    import pickle
    with open(path or MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def score(pair: str, at: datetime | None = None, bundle=None) -> dict | None:
    """Return the model's decision at a moment.

    `action` is the probability the operator would have entered here; `side` is
    the more likely direction. A signal fires only when action clears the frozen
    threshold — which is the 70th percentile of the in-sample score, chosen
    before any forward data existed and not adjustable afterwards.
    """
    import numpy as np

    bundle = bundle or load_model()
    at = at or datetime.now(timezone.utc)
    f = features(pair, at)
    if f is None:
        return None
    missing = [k for k in bundle["features"] if k not in f]
    if missing:
        raise RuntimeError(f"feature mismatch between training and live: {missing[:5]}")
    x = np.array([[float(f[k]) for k in bundle["features"]]])
    p = bundle["model"].predict_proba(bundle["scaler"].transform(x))[0]
    action = float(1 - p[0])
    side = 1 if p[1] >= p[2] else -1
    return {
        "pair": pair,
        "at_utc": at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "action": round(action, 4),
        "threshold": bundle["threshold"],
        "fires": action >= bundle["threshold"],
        "side": "long" if side > 0 else "short",
        "p_no_trade": round(float(p[0]), 4),
        "p_long": round(float(p[1]), 4),
        "p_short": round(float(p[2]), 4),
        "exit_hours": EXIT_HOURS,
    }
