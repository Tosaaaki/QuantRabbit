#!/usr/bin/env python
"""Evaluate an existing FastScalp pattern model against recent trades."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"


def _load_dataset(limit: int, min_abs_pips: float) -> tuple[np.ndarray, np.ndarray]:
    orders_db = sqlite3.connect(LOGS_DIR / "orders.db")
    trades_db = sqlite3.connect(LOGS_DIR / "trades.db")
    orders_db.row_factory = sqlite3.Row
    trades_db.row_factory = sqlite3.Row

    rows = trades_db.execute(
        """
        SELECT id, ticket_id, pl_pips
        FROM trades
        WHERE pocket='scalp_fast'
        AND pl_pips IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    features = []
    labels = []

    for row in rows:
        ticket = row["ticket_id"]
        pl_pips = float(row["pl_pips"])
        order_row = orders_db.execute(
            """
            SELECT side, request_json
            FROM orders
            WHERE ticket_id=? AND status='submit_attempt'
            ORDER BY id DESC
            LIMIT 1
            """,
            (ticket,),
        ).fetchone()
        if not order_row or order_row["request_json"] is None:
            continue
        try:
            payload = json.loads(order_row["request_json"])
        except json.JSONDecodeError:
            continue
        thesis = (payload.get("meta") or {}).get("entry_thesis") or {}
        feat = thesis.get("pattern_features")
        if not feat:
            continue
        try:
            vec = [float(x) for x in feat]
        except Exception:
            continue
        vec.append(1.0 if order_row["side"] == "buy" else -1.0)
        features.append(vec)
        labels.append(1 if pl_pips >= min_abs_pips else 0)

    orders_db.close()
    trades_db.close()

    if not features:
        raise RuntimeError("No pattern feature rows found. Collect more live data first.")

    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to trained joblib model")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-positive-pips", type=float, default=0.2)
    args = parser.parse_args()

    model = joblib.load(args.model)
    X, y = _load_dataset(args.limit, args.min_positive_pips)
    y_pred = model.predict(X)

    try:
        y_score = model.predict_proba(X)[:, 1]
    except Exception:
        y_score = None

    print(classification_report(y, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    if y_score is not None:
        try:
            auc = roc_auc_score(y, y_score)
            print(f"ROC AUC: {auc:.3f}")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
