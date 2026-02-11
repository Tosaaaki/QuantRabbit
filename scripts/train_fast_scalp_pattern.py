#!/usr/bin/env python
"""Train a lightweight classifier for FastScalp tick patterns."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    if len(set(labels)) < 2:
        raise RuntimeError("Pattern dataset has only one class. Adjust --min-positive-pips or collect more data.")

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y


def train_model(limit: int, min_abs_pips: float) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    X, y = _load_dataset(limit, min_abs_pips)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=300,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    matrix = confusion_matrix(y_test, y_pred)

    print("=== Pattern classifier report ===")
    print(report)
    print("Confusion matrix:")
    print(matrix)
    print(f"Positive rate in dataset: {mean(y):.3f}")

    return pipeline, X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fast_scalp_pattern.joblib"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-positive-pips", type=float, default=0.2)
    args = parser.parse_args()

    model, _, _ = train_model(args.limit, args.min_positive_pips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
