#!/usr/bin/env python3
"""Publish range_mode_active metric from cached factors or latest candles."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from analysis.range_guard import detect_range_mode
from indicators.calc_core import IndicatorEngine
from indicators.factor_cache import all_factors
from utils.market_hours import is_market_open
from utils.metrics_logger import log_metric


DEFAULT_MIN_INTERVAL_SEC = int(os.getenv("RANGE_MODE_PUBLISH_MIN_INTERVAL_SEC", "180"))
DEFAULT_MAX_DATA_AGE_SEC = int(os.getenv("RANGE_MODE_PUBLISH_MAX_DATA_AGE_SEC", "900"))
METRICS_DB = Path(os.getenv("RANGE_MODE_METRICS_DB", "logs/metrics.db"))
M1_PATH = Path(os.getenv("RANGE_MODE_CANDLES_M1", "logs/oanda/candles_M1_latest.json"))
H4_PATH = Path(os.getenv("RANGE_MODE_CANDLES_H4", "logs/oanda/candles_H4_latest.json"))
H1_PATH = Path(os.getenv("RANGE_MODE_CANDLES_H1", "logs/oanda/candles_H1_latest.json"))
MACRO_TF = os.getenv("RANGE_MODE_MACRO_TF", "H4").upper()
REFRESH_ENABLED = os.getenv("RANGE_MODE_PUBLISH_REFRESH", "1").lower() not in {
    "0",
    "false",
    "off",
}
ALLOW_CLOSED_STALE = os.getenv("RANGE_MODE_PUBLISH_ALLOW_CLOSED", "1").lower() not in {
    "0",
    "false",
    "off",
}


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


REFRESH_M1_COUNT = _parse_int(os.getenv("RANGE_MODE_PUBLISH_REFRESH_M1_COUNT", "500"), 500)
REFRESH_MACRO_COUNT = _parse_int(os.getenv("RANGE_MODE_PUBLISH_REFRESH_MACRO_COUNT", "200"), 200)


def _utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


def _macro_tf_seconds() -> int:
    tf = str(MACRO_TF or "").upper()
    if tf == "H1":
        return 3600
    if tf == "H4":
        return 4 * 3600
    return max(DEFAULT_MAX_DATA_AGE_SEC, 900)


def _macro_max_data_age_sec() -> int:
    explicit = _parse_int(os.getenv("RANGE_MODE_PUBLISH_MACRO_MAX_DATA_AGE_SEC", "0"), 0)
    if explicit > 0:
        return explicit
    # Macro candles are inherently older than M1 ticks; allow up to 135% of one macro bar.
    auto_limit = int(_macro_tf_seconds() * 1.35)
    return max(DEFAULT_MAX_DATA_AGE_SEC, auto_limit)


def _parse_ts(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.utcfromtimestamp(float(value))
        except Exception:
            return None
    text = str(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _latest_metric_ts(metric: str) -> Optional[dt.datetime]:
    if not METRICS_DB.exists():
        return None
    con = sqlite3.connect(str(METRICS_DB))
    try:
        row = con.execute(
            "SELECT ts FROM metrics WHERE metric = ? ORDER BY ts DESC LIMIT 1",
            (metric,),
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return _parse_ts(row[0])


def _load_candles(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        items = payload.get("candles") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [c for c in items if isinstance(c, dict)]


def _latest_candle_ts(candles: list[dict]) -> Optional[dt.datetime]:
    if not candles:
        return None
    last = candles[-1]
    return _parse_ts(last.get("time") or last.get("timestamp") or last.get("ts"))


def _age_seconds(now: dt.datetime, ts: Optional[dt.datetime]) -> Optional[float]:
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds())


def _candles_to_factors(candles: list[dict]) -> Optional[Dict[str, float]]:
    rows = []
    for c in candles:
        try:
            rows.append(
                {
                    "open": float(c.get("open")),
                    "high": float(c.get("high")),
                    "low": float(c.get("low")),
                    "close": float(c.get("close")),
                }
            )
        except Exception:
            continue
    if len(rows) < 20:
        return None
    df = pd.DataFrame(rows)
    factors = IndicatorEngine.compute(df)
    factors["timestamp"] = (
        candles[-1].get("time") or candles[-1].get("timestamp") or candles[-1].get("ts")
    )
    return factors


def _has_required(factors: Dict[str, Any]) -> bool:
    if not factors:
        return False
    required = ("adx", "bbw", "atr_pips", "vol_5m", "ma10", "ma20")
    for key in required:
        val = factors.get(key)
        if val is None:
            return False
        try:
            float(val)
        except Exception:
            return False
    return True


def _resolve_factors() -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], str, Optional[dt.datetime], Optional[dt.datetime]]:
    factors = all_factors()
    fac_m1 = factors.get("M1") or {}
    fac_macro = factors.get(MACRO_TF) or {}
    ts_m1 = _parse_ts(fac_m1.get("timestamp"))
    ts_macro = _parse_ts(fac_macro.get("timestamp"))
    if _has_required(fac_m1) and _has_required(fac_macro):
        return fac_m1, fac_macro, "factor_cache", ts_m1, ts_macro

    m1_candles = _load_candles(M1_PATH)
    macro_path = H4_PATH if MACRO_TF == "H4" else H1_PATH
    macro_candles = _load_candles(macro_path)
    fac_m1 = _candles_to_factors(m1_candles)
    fac_macro = _candles_to_factors(macro_candles)
    ts_m1 = _latest_candle_ts(m1_candles)
    ts_macro = _latest_candle_ts(macro_candles)
    return fac_m1, fac_macro, "candles", ts_m1, ts_macro


def _freshness_state(
    now: dt.datetime,
    ts_m1: Optional[dt.datetime],
    ts_macro: Optional[dt.datetime],
) -> tuple[bool, Optional[float], Optional[float], int]:
    m1_age_sec = _age_seconds(now, ts_m1)
    macro_age_sec = _age_seconds(now, ts_macro)
    macro_limit_sec = _macro_max_data_age_sec()

    stale = False
    if m1_age_sec is not None and m1_age_sec > DEFAULT_MAX_DATA_AGE_SEC:
        stale = True
    if macro_age_sec is not None and macro_age_sec > macro_limit_sec:
        stale = True
    return stale, m1_age_sec, macro_age_sec, macro_limit_sec


def _refresh_candles() -> bool:
    if not REFRESH_ENABLED:
        return False
    macro_tf = MACRO_TF if MACRO_TF in {"H4", "H1"} else "H4"
    macro_count = REFRESH_MACRO_COUNT
    m1_count = REFRESH_M1_COUNT
    try:
        logging.info("[range_metric] refreshing candles (M1=%d %s=%d)", m1_count, macro_tf, macro_count)
        subprocess.run(
            [
                sys.executable,
                "scripts/refresh_latest_candles.py",
                "--granularity",
                "M1",
                "--count",
                str(m1_count),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/refresh_latest_candles.py",
                "--granularity",
                macro_tf,
                "--count",
                str(macro_count),
            ],
            check=True,
        )
        return True
    except Exception as exc:
        logging.warning("[range_metric] refresh failed: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish range_mode_active metric.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--force", action="store_true", help="Ignore min interval guard.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    enabled = os.getenv("RANGE_MODE_PUBLISH_ENABLED", "1").lower() not in {"0", "false", "off"}
    if not enabled:
        logging.info("[range_metric] disabled")
        return

    now = _utcnow()
    last_ts = _latest_metric_ts("range_mode_active")
    if not args.force and last_ts is not None and DEFAULT_MIN_INTERVAL_SEC > 0:
        delta = (now - last_ts).total_seconds()
        if delta < DEFAULT_MIN_INTERVAL_SEC:
            logging.info("[range_metric] skip (recent %.1fs)", delta)
            return

    fac_m1, fac_macro, source, ts_m1, ts_macro = _resolve_factors()
    stale_data, m1_age_sec, macro_age_sec, macro_limit_sec = _freshness_state(now, ts_m1, ts_macro)
    needs_refresh = (
        not fac_m1
        or not fac_macro
        or stale_data
    )
    if needs_refresh and _refresh_candles():
        fac_m1, fac_macro, source, ts_m1, ts_macro = _resolve_factors()
        stale_data, m1_age_sec, macro_age_sec, macro_limit_sec = _freshness_state(
            now,
            ts_m1,
            ts_macro,
        )

    if not fac_m1 or not fac_macro:
        logging.warning("[range_metric] missing factors (source=%s)", source)
        return
    stale_reason = None
    if stale_data:
        if ALLOW_CLOSED_STALE and not is_market_open(now):
            stale_reason = "market_closed"
        else:
            logging.warning(
                "[range_metric] data too old (m1_age=%.1fs macro_age=%.1fs m1_limit=%ss macro_limit=%ss)",
                -1.0 if m1_age_sec is None else m1_age_sec,
                -1.0 if macro_age_sec is None else macro_age_sec,
                DEFAULT_MAX_DATA_AGE_SEC,
                macro_limit_sec,
            )
            return

    range_ctx = detect_range_mode(fac_m1, fac_macro, env_tf="M1", macro_tf=MACRO_TF)
    tags = {
        "reason": range_ctx.reason or "unknown",
        "score": round(float(range_ctx.score), 3),
        "source": source,
        "env_tf": "M1",
        "macro_tf": MACRO_TF,
        "m1_limit_sec": DEFAULT_MAX_DATA_AGE_SEC,
        "macro_limit_sec": macro_limit_sec,
    }
    if m1_age_sec is not None:
        tags["age_sec"] = int(m1_age_sec)
        tags["m1_age_sec"] = int(m1_age_sec)
    if macro_age_sec is not None:
        tags["macro_age_sec"] = int(macro_age_sec)
    if stale_reason:
        tags["stale"] = stale_reason
    written = log_metric("range_mode_active", 1.0 if range_ctx.active else 0.0, tags=tags, ts=now)
    if not written:
        logging.warning(
            "[range_metric] metric_write_failed metric=range_mode_active db=%s",
            os.getenv("METRICS_DB_PATH", "logs/metrics.db"),
        )
        return
    logging.info(
        "[range_metric] logged active=%s source=%s reason=%s",
        range_ctx.active,
        source,
        range_ctx.reason,
    )


if __name__ == "__main__":
    main()
