from __future__ import annotations

"""
analysis.kaizen
~~~~~~~~~~~~~~~~

Continuous retrospective and tuning:
- Persist exit policy per (strategy, pocket)
- Audit closed trades to compute MFE/MAE from OANDA M1 candles
- Detect "gave back profit" patterns and auto‑adjust BE/trailing

Tables (SQLite logs/trades.db):
- exit_policy(strategy, pocket, be_trigger_pips, trail_atr_mult, min_trail_pips,
              meanrev_max_min, meanrev_rsi_exit, updated_at)
- trade_metrics(ticket_id PRIMARY KEY, mfe_pips, mae_pips, review_tag, computed_at)
"""

import asyncio
import datetime as dt
import json
import pathlib
import sqlite3
from typing import Dict, Tuple, Any, List

import requests

from utils.secrets import get_secret
import pandas as pd

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover
    firestore = None  # type: ignore


DB = pathlib.Path("logs/trades.db")
DB.parent.mkdir(exist_ok=True)
con = sqlite3.connect(DB)


def _ensure_tables() -> None:
    cur = con.cursor()
    # trades table might be created elsewhere; ensure existence for queries
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY,
          ticket_id TEXT UNIQUE,
          pocket TEXT,
          instrument TEXT,
          units INTEGER,
          entry_price REAL,
          close_price REAL,
          pl_pips REAL,
          entry_time TEXT,
          close_time TEXT,
          strategy TEXT,
          macro_regime TEXT,
          micro_regime TEXT,
          close_reason TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS exit_policy (
          strategy TEXT,
          pocket   TEXT,
          be_trigger_pips REAL,
          trail_atr_mult  REAL,
          min_trail_pips  REAL,
          meanrev_max_min REAL,
          meanrev_rsi_exit REAL,
          updated_at TEXT,
          PRIMARY KEY(strategy, pocket)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_metrics (
          ticket_id TEXT PRIMARY KEY,
          mfe_pips REAL,
          mae_pips REAL,
          review_tag TEXT,
          computed_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS exit_advice_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          recorded_at TEXT,
          trade_id TEXT,
          strategy TEXT,
          pocket TEXT,
          version TEXT,
          event_type TEXT,
          action TEXT,
          confidence REAL,
          target_tp_pips REAL,
          target_sl_pips REAL,
          note TEXT,
          price REAL,
          move_pips REAL,
          payload_json TEXT
        )
        """
    )
    con.commit()


_ensure_tables()

# Firestore helper (optional in Cloud Run)
def _fs() -> Any | None:
    if firestore is None:
        return None
    try:
        return firestore.Client()
    except Exception:
        return None


DEFAULTS = {
    ("TrendMA", "micro"): dict(be=10.0, atr=1.0, mintrail=8.0),
    ("TrendMA", "macro"): dict(be=20.0, atr=1.5, mintrail=15.0),
    ("Donchian55", "micro"): dict(be=10.0, atr=1.0, mintrail=8.0),
    ("Donchian55", "macro"): dict(be=20.0, atr=1.5, mintrail=15.0),
    ("BB_RSI", "micro"): dict(be=8.0, atr=0.8, mintrail=6.0),
    ("BB_RSI", "macro"): dict(be=12.0, atr=1.2, mintrail=12.0),
    ("NewsSpikeReversal", "micro"): dict(be=7.0, atr=1.0, mintrail=10.0),
    ("NewsSpikeReversal", "macro"): dict(be=10.0, atr=1.2, mintrail=12.0),
}


def get_policy(strategy: str, pocket: str) -> Dict[str, float]:
    """Fetch exit policy with precedence: Firestore -> SQLite -> defaults.
    Ensures a policy exists by writing defaults back to the chosen store if missing.
    """
    # 1) Firestore
    fs = _fs()
    if fs is not None:
        try:
            doc_id = f"{strategy}_{pocket}"
            ref = fs.collection("exit_policy").document(doc_id)
            snap = ref.get()
            if snap.exists:
                data = snap.to_dict() or {}
                return {
                    "be_trigger_pips": float(data.get("be_trigger_pips", 10.0 if pocket == "micro" else 20.0)),
                    "trail_atr_mult": float(data.get("trail_atr_mult", 1.0 if pocket == "micro" else 1.5)),
                    "min_trail_pips": float(data.get("min_trail_pips", 8.0 if pocket == "micro" else 15.0)),
                    "meanrev_max_min": float(data.get("meanrev_max_min", 30.0)),
                    "meanrev_rsi_exit": float(data.get("meanrev_rsi_exit", 50.0)),
                }
        except Exception:
            pass

    # 2) SQLite fallback
    try:
        cur = con.cursor()
        row = cur.execute(
            "SELECT be_trigger_pips, trail_atr_mult, min_trail_pips, meanrev_max_min, meanrev_rsi_exit FROM exit_policy WHERE strategy=? AND pocket=?",
            (strategy, pocket),
        ).fetchone()
        if row:
            be, atr, mintrail, mrmax, mrrsi = row
            return {
                "be_trigger_pips": float(be),
                "trail_atr_mult": float(atr),
                "min_trail_pips": float(mintrail),
                "meanrev_max_min": float(mrmax if mrmax is not None else 30.0),
                "meanrev_rsi_exit": float(mrrsi if mrrsi is not None else 50.0),
            }
    except Exception:
        pass

    # 3) Defaults and write-back
    base = DEFAULTS.get((strategy, pocket)) or DEFAULTS.get(("TrendMA", pocket)) or {
        "be": 10.0,
        "atr": 1.0,
        "mintrail": 8.0,
    }
    payload = {
        "be_trigger_pips": float(base["be"]),
        "trail_atr_mult": float(base["atr"]),
        "min_trail_pips": float(base["mintrail"]),
        "meanrev_max_min": 30.0,
        "meanrev_rsi_exit": 50.0,
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "strategy": strategy,
        "pocket": pocket,
    }
    # write to Firestore if available
    if _fs() is not None:
        try:
            _fs().collection("exit_policy").document(f"{strategy}_{pocket}").set(payload)
            return get_policy(strategy, pocket)
        except Exception:
            pass
    # write to SQLite as last resort
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO exit_policy(strategy,pocket,be_trigger_pips,trail_atr_mult,min_trail_pips,meanrev_max_min,meanrev_rsi_exit,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                strategy,
                pocket,
                payload["be_trigger_pips"],
                payload["trail_atr_mult"],
                payload["min_trail_pips"],
                payload["meanrev_max_min"],
                payload["meanrev_rsi_exit"],
                payload["updated_at"],
            ),
        )
        con.commit()
    except Exception:
        pass
    return {
        "be_trigger_pips": payload["be_trigger_pips"],
        "trail_atr_mult": payload["trail_atr_mult"],
        "min_trail_pips": payload["min_trail_pips"],
        "meanrev_max_min": payload["meanrev_max_min"],
        "meanrev_rsi_exit": payload["meanrev_rsi_exit"],
    }


def _oanda_env() -> Tuple[str, Dict[str, str]] | None:
    try:
        token = get_secret("oanda_token")
        try:
            pract = get_secret("oanda_practice").lower() == "true"
        except Exception:
            pract = True
        host = (
            "https://api-fxpractice.oanda.com"
            if pract
            else "https://api-fxtrade.oanda.com"
        )
        headers = {"Authorization": f"Bearer {token}"}
        return host, headers
    except Exception:
        return None


def _fetch_m1_candles(from_iso: str, to_iso: str) -> List[Dict[str, Any]]:
    env = _oanda_env()
    if not env:
        return []
    REST_HOST, HEADERS = env
    instrument = "USD_JPY"
    url = f"{REST_HOST}/v3/instruments/{instrument}/candles"
    params = {
        "granularity": "M1",
        "price": "M",
        "from": from_iso,
        "to": to_iso,
        "count": 2000,
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=7)
        r.raise_for_status()
        return r.json().get("candles", [])
    except requests.RequestException:
        return []


def _mfe_mae(entry: float, direction: int, candles: List[Dict[str, Any]]) -> Tuple[float, float]:
    pip = 0.01
    highs: List[float] = []
    lows: List[float] = []
    for c in candles:
        mid = c.get("mid") or {}
        try:
            h = float(mid.get("h"))
            l = float(mid.get("l"))
        except Exception:
            # some APIs nest differently; try top-level h/l
            h = float(c.get("h", 0.0))
            l = float(c.get("l", 0.0))
        highs.append(h)
        lows.append(l)
    if not highs or not lows:
        return 0.0, 0.0
    if direction > 0:
        mfe = max(0.0, (max(highs) - entry) / pip)
        mae = max(0.0, (entry - min(lows)) / pip)
    else:
        mfe = max(0.0, (entry - min(lows)) / pip)
        mae = max(0.0, (max(highs) - entry) / pip)
    return float(mfe), float(mae)


def audit_recent_trades(days: int = 2) -> None:
    """Audit recent closed trades, compute MFE/MAE, tag review, and store metrics.
    Prefers Firestore trades; falls back to local SQLite if Firestore unavailable.
    """
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat(timespec="seconds")
    trades: List[Tuple[str, int, float, str, str, float, str, str]] = []
    fs = _fs()
    if fs is not None:
        try:
            # Avoid composite index requirement: order by 'ts' only and filter in app code
            q = fs.collection("trades").order_by("ts", direction=firestore.Query.DESCENDING).limit(800)
            for d in q.stream():
                x = d.to_dict() or {}
                if (x.get("instrument") or "") != "USD_JPY":
                    continue
                if (x.get("state") or "").upper() != "CLOSED":
                    continue
                ct = x.get("close_time") or x.get("fill_time") or x.get("ts") or ""
                if ct and str(ct) < since:
                    continue
                ticket = str(x.get("trade_id") or "")
                try:
                    units = int(x.get("units") or 0)
                    entry = float(x.get("fill_price") or x.get("price") or 0.0)
                    pl = float(x.get("pl_pips") or x.get("realized_pl") or 0.0)
                except Exception:
                    continue
                et = x.get("fill_time") or x.get("ts") or ""
                strat = x.get("strategy") or ""
                pocket = x.get("pocket") or ("macro" if abs(units) >= 100000 else "micro")
                trades.append((ticket, units, entry, str(et), str(ct), pl, strat, pocket))
        except Exception:
            trades = []
    if not trades:
        # fallback to sqlite copy if exists
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT ticket_id, units, entry_price, entry_time, close_time, pl_pips, strategy, pocket
            FROM trades
            WHERE close_time >= ? AND instrument='USD_JPY'
            """,
            (since,),
        ).fetchall()
        trades = rows or []

    # load already processed ticket ids (Firestore preferred)
    done: set[str] = set()
    if _fs() is not None:
        try:
            for d in _fs().collection("trade_metrics").limit(1000).stream():
                x = d.to_dict() or {}
                tid = str(x.get("ticket_id") or d.id)
                done.add(tid)
        except Exception:
            pass
    if not done:
        cur = con.cursor()
        try:
            done = {r[0] for r in cur.execute("SELECT ticket_id FROM trade_metrics").fetchall()}
        except Exception:
            done = set()

    for (ticket, units, entry, et, ct, pl, strat, pocket) in trades:
        if not ticket or ticket in done:
            continue
        try:
            direction = 1 if int(units) > 0 else -1
        except Exception:
            direction = 1
        # Fetch M1 window a bit wider
        e = dt.datetime.fromisoformat(str(et).replace("Z", "+00:00")) - dt.timedelta(minutes=1)
        c = dt.datetime.fromisoformat(str(ct).replace("Z", "+00:00")) + dt.timedelta(minutes=1)
        candles = _fetch_m1_candles(e.isoformat(), c.isoformat())
        mfe, mae = _mfe_mae(float(entry), direction, candles)
        # Review tag: gave_back if loss but had >=10 pips MFE
        tag = ""
        try:
            if float(pl) < 0.0 and mfe >= 10.0:
                tag = "gave_back"
            elif float(pl) > 0.0 and mae >= 10.0:
                tag = "held_through_pain"
        except Exception:
            tag = ""
        ts_now = dt.datetime.utcnow().isoformat(timespec="seconds")
        # write to Firestore if available
        if _fs() is not None:
            try:
                _fs().collection("trade_metrics").document(str(ticket)).set(
                    {
                        "ticket_id": str(ticket),
                        "mfe_pips": float(mfe),
                        "mae_pips": float(mae),
                        "review_tag": tag,
                        "computed_at": ts_now,
                    }
                )
            except Exception:
                pass
        # also write to SQLite for local analysis
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO trade_metrics(ticket_id, mfe_pips, mae_pips, review_tag, computed_at) VALUES (?,?,?,?,?)",
                (ticket, float(mfe), float(mae), tag, ts_now),
            )
            con.commit()
        except Exception:
            pass


def _adjust_policy(strategy: str, pocket: str, *, decrease: bool) -> None:
    cur = con.cursor()
    pol = get_policy(strategy, pocket)
    be = pol["be_trigger_pips"]
    atr = pol["trail_atr_mult"]
    mintrail = pol["min_trail_pips"]

    # Bounds
    min_be = 5.0 if pocket == "micro" else 10.0
    max_be = 30.0 if pocket == "micro" else 40.0
    min_atr = 0.8 if pocket == "micro" else 1.2
    max_atr = 2.5
    min_mintrail = 6.0 if pocket == "micro" else 12.0
    max_mintrail = 30.0

    if decrease:
        be = max(min_be, be - 2.0)
        atr = max(min_atr, atr - 0.2)
        mintrail = max(min_mintrail, mintrail - 2.0)
    else:
        be = min(max_be, be + 2.0)
        atr = min(max_atr, atr + 0.2)
        mintrail = min(max_mintrail, mintrail + 2.0)

    ts_now = dt.datetime.utcnow().isoformat(timespec="seconds")
    # Firestore write
    if _fs() is not None:
        try:
            _fs().collection("exit_policy").document(f"{strategy}_{pocket}").set(
                {
                    "strategy": strategy,
                    "pocket": pocket,
                    "be_trigger_pips": float(be),
                    "trail_atr_mult": float(atr),
                    "min_trail_pips": float(mintrail),
                    "updated_at": ts_now,
                },
                merge=True,
            )
        except Exception:
            pass
    # SQLite write
    try:
        cur = con.cursor()
        cur.execute(
            """
            UPDATE exit_policy
            SET be_trigger_pips=?, trail_atr_mult=?, min_trail_pips=?, updated_at=?
            WHERE strategy=? AND pocket=?
            """,
            (float(be), float(atr), float(mintrail), ts_now, strategy, pocket),
        )
        con.commit()
    except Exception:
        pass


def tune_policies(lookback_days: int = 3, min_trades: int = 6, threshold: float = 0.3) -> None:
    """If too many 'gave_back' trades, tighten exits; otherwise relax slightly."""
    cur = con.cursor()
    since = (dt.datetime.utcnow() - dt.timedelta(days=lookback_days)).isoformat(timespec="seconds")
    rows = cur.execute(
        """
        SELECT t.strategy, t.pocket, COUNT(*), SUM(CASE WHEN m.review_tag='gave_back' THEN 1 ELSE 0 END)
        FROM trades t
        JOIN trade_metrics m ON t.ticket_id=m.ticket_id
        WHERE t.close_time>=? AND t.strategy IS NOT NULL
        GROUP BY t.strategy, t.pocket
        """,
        (since,),
    ).fetchall()
    for strat, pocket, cnt, gb in rows:
        if not strat or not pocket:
            continue
        cnt = int(cnt or 0)
        gb = int(gb or 0)
        if cnt < min_trades:
            continue
        ratio = gb / cnt if cnt else 0.0
        if ratio >= threshold:
            _adjust_policy(strat, pocket, decrease=True)
        elif ratio <= 0.1:
            _adjust_policy(strat, pocket, decrease=False)


def exit_advice_summary(days: int = 7) -> Dict[str, Any]:
    """Return aggregated GPT exit advice metrics for downstream tuning."""

    query = (
        """
        SELECT e.*, t.pl_pips AS realized_pips, t.close_time
        FROM exit_advice_events e
        LEFT JOIN trades t ON t.ticket_id = e.trade_id
        """
    )
    df = pd.read_sql_query(query, con, parse_dates=["recorded_at", "close_time"])
    if df.empty:
        return {"message": "No exit advice events recorded."}

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    df = df[df["recorded_at"] >= cutoff]
    if df.empty:
        return {"message": f"No exit advice events in last {days} days."}

    summary: Dict[str, Any] = {"window_days": days}
    close_df = df[df["event_type"] == "close_request"].copy()
    if not close_df.empty:
        close_df["win"] = close_df["realized_pips"].fillna(0) > 0
        summary["close_success_rate"] = float(close_df["win"].mean())
        summary["close_avg_conf"] = float(close_df["confidence"].mean())
    adjust_df = df[df["event_type"] == "adjust"].copy()
    if not adjust_df.empty:
        summary["adjust_avg_conf"] = float(adjust_df["confidence"].mean())

    strat = (
        df.groupby(["strategy", "pocket", "event_type"])
        .agg(events=("id", "count"), avg_conf=("confidence", "mean"))
        .reset_index()
    )
    summary["by_strategy"] = strat.to_dict("records")

    suggestions: Dict[str, float] = {}
    close_rate = summary.get("close_success_rate")
    if close_rate is not None:
        if close_rate < 0.5:
            suggestions["EXIT_GPT_CLOSE_CONF"] = min(0.9, (summary.get("close_avg_conf") or 0.55) + 0.05)
        elif close_rate > 0.65:
            suggestions["EXIT_GPT_CLOSE_CONF"] = max(0.2, (summary.get("close_avg_conf") or 0.55) - 0.05)
    adjust_avg = summary.get("adjust_avg_conf")
    if adjust_avg is not None:
        if adjust_avg < 0.3:
            suggestions["EXIT_GPT_ADJUST_CONF"] = min(0.9, adjust_avg + 0.05)
        elif adjust_avg > 0.6:
            suggestions["EXIT_GPT_ADJUST_CONF"] = max(0.1, adjust_avg - 0.05)
    if suggestions:
        summary["suggestions"] = suggestions
    return summary


async def audit_loop():
    """Background loop: audit recent trades and tune policies."""
    while True:
        try:
            audit_recent_trades(days=2)
            tune_policies(lookback_days=3)
        except Exception as e:
            print(f"[kaizen] audit error: {e}")
        await asyncio.sleep(300)
