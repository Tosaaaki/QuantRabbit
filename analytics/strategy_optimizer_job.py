"""Strategy optimisation pipeline.

This module orchestrates three steps:

1. Aggregate recent performance for each pocket/strategy combination.
2. Train (or refresh) a BigQuery ML logistic regression model that predicts the
   probability of winning trades based on contextual features.
3. Derive actionable recommendations (boost / suspend / caution) and store
   them in BigQuery so that the execution engine can adjust confidence and risk
   in near real-time. Optionally, the module can call OpenAI to summarise the
   findings for human oversight.

The code is written to run inside Cloud Run / Composer, but it also works
locally as long as application default credentials are available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from google.api_core import exceptions as gexc
from google.cloud import bigquery

try:  # Optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore


DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_PROJECT = os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
RECO_TABLE = os.getenv("BQ_RECOMMENDATION_TABLE", "strategy_recommendations")
MODEL_NAME = os.getenv("BQ_STRATEGY_MODEL", "strategy_outcome_model")


def _ensure_reco_table(client: bigquery.Client, dataset_id: str) -> None:
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except gexc.NotFound:
        logging.info("[BQ] dataset %s missing. Creating...", dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.getenv("BQ_LOCATION", "US")
        client.create_dataset(dataset, exists_ok=True)

    schema = [
        bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("pocket", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("win_rate", "FLOAT64"),
        bigquery.SchemaField("profit_factor", "FLOAT64"),
        bigquery.SchemaField("action", "STRING"),
        bigquery.SchemaField("confidence_scale", "FLOAT64"),
        bigquery.SchemaField("reason", "STRING"),
        bigquery.SchemaField("notes", "STRING"),
    ]
    table_ref = dataset_ref.table(RECO_TABLE)
    try:
        client.get_table(table_ref)
    except gexc.NotFound:
        logging.info("[BQ] creating table %s", table_ref.path)
        client.create_table(bigquery.Table(table_ref, schema=schema))


def _train_model(client: bigquery.Client, dataset_id: str) -> None:
    query = f"""
    CREATE OR REPLACE MODEL `{client.project}.{dataset_id}.{MODEL_NAME}`
    OPTIONS(model_type='logistic_reg', input_label_cols=['won']) AS
    SELECT
      pocket,
      COALESCE(strategy, 'unknown') AS strategy,
      SAFE_DIVIDE(pl_pips, 100.0) AS pnl_pips,
      TIMESTAMP_DIFF(close_time, entry_time, SECOND) AS hold_seconds,
      CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END AS won
    FROM `{client.project}.{dataset_id}.trades_raw`
    WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
      AND state = 'CLOSED'
"""
    client.query(query).result()


def _fetch_latest_metrics(client: bigquery.Client, dataset_id: str) -> List[bigquery.table.Row]:
    query = f"""
    SELECT
      pocket,
      COALESCE(strategy, 'unknown') AS strategy,
      COUNT(*) AS trades,
      AVG(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win_rate,
      SAFE_DIVIDE(
        SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END),
        NULLIF(ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)), 0)
      ) AS profit_factor
    FROM `{client.project}.{dataset_id}.trades_raw`
    WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
      AND state = 'CLOSED'
    GROUP BY pocket, strategy
"""
    return list(client.query(query).result())


def _derive_actions(rows: Iterable[bigquery.table.Row]) -> List[Dict[str, object]]:
    out = []
    now = datetime.now(timezone.utc)
    for row in rows:
        trades = int(row["trades"] or 0)
        win_rate = float(row["win_rate"] or 0.0)
        pf = float(row["profit_factor"] or 0.0)
        action = "hold"
        confidence = 1.0
        reason = None

        if trades >= 8 and (win_rate < 0.42 or pf < 0.85):
            action = "suspend"
            confidence = 0.0
            reason = "underperforming"
        elif trades >= 5 and (win_rate < 0.5 or pf < 0.95):
            action = "caution"
            confidence = 0.5
            reason = "soft_patch"
        elif trades >= 5 and win_rate > 0.62 and pf > 1.15:
            action = "boost"
            confidence = 1.2
            reason = "strong_performance"

        out.append(
            {
                "generated_at": now,
                "pocket": row["pocket"],
                "strategy": row["strategy"],
                "win_rate": win_rate,
                "profit_factor": pf,
                "action": action,
                "confidence_scale": confidence,
                "reason": reason,
            }
        )
    return out


def _insert_recommendations(
    client: bigquery.Client,
    dataset_id: str,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        return
    table_ref = client.dataset(dataset_id).table(RECO_TABLE)
    payload = []
    for row in rows:
        payload.append({**row, "generated_at": row["generated_at"].isoformat()})
    errors = client.insert_rows_json(table_ref, payload)
    if errors:
        raise RuntimeError(f"BigQuery insert failed: {errors}")


def _summarise_with_gpt(rows: List[Dict[str, object]]) -> str | None:
    if not rows or OpenAI is None:
        return None
    try:  # pragma: no cover - relies on external API
        client = OpenAI()
    except Exception:
        return None

    bullet_lines = [
        f"- {r['pocket']}/{r['strategy']}: action={r['action']} win_rate={r['win_rate']:.2f} pf={r['profit_factor']:.2f}"
        for r in rows
    ]
    prompt = (
        "以下の指標を分析し、戦略ポートフォリオの調整ポイントを100文字以内でまとめてください:\n"
        + "\n".join(bullet_lines)
    )
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL_OPTIMIZER", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=180,
        )
        return response.output_text.strip()
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[GPT] optimisation summary failed: %s", exc)
        return None


def run(project: str | None = None, dataset: str = DEFAULT_DATASET) -> None:
    client = bigquery.Client(project=project) if project else bigquery.Client()
    _ensure_reco_table(client, dataset)

    logging.info("[OPTIM] training BigQuery ML model...")
    _train_model(client, dataset)

    logging.info("[OPTIM] deriving recommendations...")
    metrics = _fetch_latest_metrics(client, dataset)
    actions = _derive_actions(metrics)
    summary = _summarise_with_gpt(actions)
    if summary:
        logging.info("[OPTIM] summary: %s", summary)
        for row in actions:
            row["notes"] = summary

    _insert_recommendations(client, dataset, actions)
    logging.info("[OPTIM] done rows=%s", len(actions))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strategy optimiser job")
    parser.add_argument("--project", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(project=args.project, dataset=args.dataset)


if __name__ == "__main__":
    main()
