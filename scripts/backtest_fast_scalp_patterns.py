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
    trades_db = sqlite3.connect(LOGS_DIR / "trades.db")
    trades_db.row_factory = sqlite3.Row

    rows = trades_db.execute(
        """
        SELECT id, units, pl_pips, entry_thesis
        FROM trades
        WHERE pocket='scalp_fast'
        AND pl_pips IS NOT NULL
        AND entry_thesis IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    features = []
    labels = []

    for row in rows:
        units = int(row["units"] or 0)
        pl_pips = float(row["pl_pips"])
        try:
            thesis = json.loads(row["entry_thesis"])
        except json.JSONDecodeError:
            continue
        feat = thesis.get("pattern_features")
        if not feat:
            continue
        try:
            vec = [float(x) for x in feat]
        except Exception:
            continue
        vec.append(1.0 if units > 0 else -1.0)
        features.append(vec)
        labels.append(1 if pl_pips >= min_abs_pips else 0)

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
