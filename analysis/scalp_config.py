from __future__ import annotations

"""Helpers to manage scalping configuration overrides."""

import sqlite3
import pathlib
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Dict, Any

try:
    from google.cloud import firestore
except Exception:  # pragma: no cover - Firestore not available
    firestore = None  # type: ignore

_DB_PATH = pathlib.Path("logs/trades.db")
_COLLECTION = "config"
_DOCUMENT = "scalp"
_DEFAULT_SHARE = 0.03
_DEFAULT_BASE_LOT = 0.008


def _firestore_client():
    if firestore is None:  # pragma: no cover
        return None
    try:
        return firestore.Client()
    except Exception:
        return None


def load_overrides() -> Dict[str, Any]:
    client = _firestore_client()
    if client is None:
        return {}
    try:
        doc = client.collection(_COLLECTION).document(_DOCUMENT).get()
        if doc.exists:
            return doc.to_dict() or {}
    except Exception:
        return {}
    return {}


def store_overrides(data: Dict[str, Any]) -> None:
    client = _firestore_client()
    if client is None:
        return
    try:
        payload = dict(data)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        client.collection(_COLLECTION).document(_DOCUMENT).set(payload)
    except Exception:
        pass


def compute_initial_from_trades(lookback_days: int = 30) -> Dict[str, float]:
    if not _DB_PATH.exists():
        return {"share": _DEFAULT_SHARE, "base_lot": _DEFAULT_BASE_LOT}
    try:
        con = sqlite3.connect(_DB_PATH)
    except Exception:
        return {"share": _DEFAULT_SHARE, "base_lot": _DEFAULT_BASE_LOT}

    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat(timespec="seconds")
    try:
        rows = con.execute(
            "SELECT pocket, ABS(CAST(units AS REAL)) FROM trades WHERE close_time >= ? AND units IS NOT NULL",
            (cutoff,),
        ).fetchall()
    except Exception:
        con.close()
        return {"share": _DEFAULT_SHARE, "base_lot": _DEFAULT_BASE_LOT}
    finally:
        con.close()

    if not rows:
        return {"share": _DEFAULT_SHARE, "base_lot": _DEFAULT_BASE_LOT}

    lot_samples = []
    pocket_totals: Dict[str, float] = {}
    for pocket, units in rows:
        try:
            val = float(units)
        except (TypeError, ValueError):
            continue
        lot = val / 100000.0
        if lot <= 0:
            continue
        pocket_totals[pocket or "unknown"] = pocket_totals.get(pocket or "unknown", 0.0) + lot
        if pocket == "micro":
            lot_samples.append(lot)

    if lot_samples:
        base_lot = max(min(median(lot_samples), 0.2), 0.005)
    else:
        base_lot = _DEFAULT_BASE_LOT

    total = sum(pocket_totals.values())
    micro_total = pocket_totals.get("micro", 0.0)
    if total > 0:
        share = micro_total / total * 0.4
        share = max(0.02, min(share, 0.15))
    else:
        share = _DEFAULT_SHARE

    return {"share": round(share, 3), "base_lot": round(base_lot, 3)}


def ensure_overrides() -> Dict[str, Any]:
    overrides = load_overrides()
    if overrides:
        return overrides
    initial = compute_initial_from_trades()
    store_overrides(initial)
    return initial
