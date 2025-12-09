from __future__ import annotations

"""
BigQuery 上の trades テーブルを集計し、戦略ごとのスコア/ロット係数/SLTP 推奨値を
Firestore にコンパクトに書き出すヘルパ。

意図:
- リアルタイム売買のホットパスに載せず、外部ジョブから Firestore の 1 ドキュメントへ
  サマリを書き込む（VM 側は読み取りキャッシュのみ）。
- テーブルの列が不足していても動作するように、存在チェックとフォールバックを入れている。
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from google.api_core import exceptions as gexc
from google.cloud import bigquery

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    firestore = None

_DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
_DEFAULT_TABLE = os.getenv("BQ_TRADES_TABLE", "trades_raw")
_DEFAULT_PROJECT = (
    os.getenv("BQ_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("GCP_PROJECT")
    or None
)


@dataclass
class StrategyScore:
    strategy: str
    pocket: str
    regime: str
    trade_count: int
    win_rate: float
    profit_factor: float
    sharpe_like: float
    avg_pips: float
    std_pips: float
    lot_multiplier: float
    score: float
    tp_pips: Optional[float]
    sl_pips: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "pocket": self.pocket,
            "regime": self.regime,
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_like": round(self.sharpe_like, 4),
            "avg_pips": round(self.avg_pips, 4),
            "std_pips": round(self.std_pips, 4),
            "lot_multiplier": round(self.lot_multiplier, 4),
            "score": round(self.score, 4),
            "tp_pips": self._round_or_none(self.tp_pips),
            "sl_pips": self._round_or_none(self.sl_pips),
        }

    @staticmethod
    def _round_or_none(val: Optional[float]) -> Optional[float]:
        return round(float(val), 4) if val is not None and math.isfinite(val) else None


class StrategyScoreExporter:
    """
    - BigQuery から戦略パフォーマンスを集計
    - ロット係数/SLTP 推奨値を算出
    - Firestore に compact JSON を上書き（オプション）
    """

    def __init__(
        self,
        *,
        project_id: Optional[str] = _DEFAULT_PROJECT,
        dataset_id: str = _DEFAULT_DATASET,
        trades_table: str = _DEFAULT_TABLE,
        lookback_days: int = 14,
        min_trades: int = 15,
        firestore_collection: str = "strategy_scores",
        firestore_document: str = "current",
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.trades_table = trades_table
        self.lookback_days = lookback_days
        self.min_trades = min_trades
        self.fs_collection = firestore_collection
        self.fs_document = firestore_document
        self.bq_client = bigquery.Client(project=self.project_id)
        self._table_schema = self._get_table_schema()

    # -------------------- public API --------------------

    def build_payload(self) -> Dict[str, Any]:
        rows = list(self._query())
        scores = [self._build_score(r) for r in rows]
        scores = [s for s in scores if s is not None]  # type: ignore[list-item]
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "project": self.bq_client.project,
            "dataset": self.dataset_id,
            "table": self.trades_table,
            "lookback_days": self.lookback_days,
            "min_trades": self.min_trades,
            "strategy_scores": [s.as_dict() for s in scores],
            "count": len(scores),
        }
        return payload

    def push_firestore(self, payload: Dict[str, Any]) -> None:
        if firestore is None:
            raise RuntimeError("google-cloud-firestore is not installed")
        client = firestore.Client(project=self.project_id)
        doc_ref = client.collection(self.fs_collection).document(self.fs_document)
        doc_ref.set(payload)
        logging.info(
            "[FS] wrote %d strategy scores to %s/%s",
            payload.get("count", 0),
            self.fs_collection,
            self.fs_document,
        )

    # -------------------- internal helpers --------------------

    def _get_table_schema(self) -> Dict[str, bigquery.SchemaField]:
        table_ref = f"{self.bq_client.project}.{self.dataset_id}.{self.trades_table}"
        try:
            table = self.bq_client.get_table(table_ref)
            return {f.name: f for f in table.schema}
        except Exception as exc:
            logging.warning("[BQ] failed to fetch schema for %s: %s", table_ref, exc)
            return {}

    def _has_column(self, name: str) -> bool:
        return name in self._table_schema

    def _query(self) -> Iterable[bigquery.table.Row]:
        table_ref = f"{self.bq_client.project}.{self.dataset_id}.{self.trades_table}"
        strategy_expr = "'unknown'"
        if self._has_column("strategy") and self._has_column("strategy_tag"):
            strategy_expr = "COALESCE(SAFE_CAST(strategy AS STRING), SAFE_CAST(strategy_tag AS STRING), 'unknown')"
        elif self._has_column("strategy"):
            strategy_expr = "COALESCE(SAFE_CAST(strategy AS STRING), 'unknown')"
        elif self._has_column("strategy_tag"):
            strategy_expr = "COALESCE(SAFE_CAST(strategy_tag AS STRING), 'unknown')"

        regime_expr = "'all'"
        if self._has_column("regime"):
            regime_expr = "COALESCE(SAFE_CAST(regime AS STRING), 'all')"
        pocket_expr = "COALESCE(SAFE_CAST(pocket AS STRING), 'unknown')"

        sql = f"""
WITH base AS (
  SELECT
    {pocket_expr} AS pocket,
    {strategy_expr} AS strategy,
    {regime_expr} AS regime,
    SAFE_CAST(pl_pips AS FLOAT64) AS pl_pips,
    SAFE_CAST(units AS INT64) AS units,
    SAFE_CAST(entry_time AS STRING) AS entry_time,
    SAFE_CAST(close_time AS STRING) AS close_time
  FROM `{table_ref}`
  WHERE state = 'CLOSED'
    AND close_time IS NOT NULL
    AND updated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
    AND pl_pips IS NOT NULL
)
SELECT
  pocket,
  strategy,
  regime,
  COUNT(*) AS trade_count,
  SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS winning_trades,
  SUM(pl_pips) AS total_pips,
  AVG(pl_pips) AS avg_pips,
  STDDEV_SAMP(pl_pips) AS std_pips,
  SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS gross_profit,
  ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)) AS gross_loss,
  AVG(
    CASE
      WHEN close_time IS NOT NULL AND entry_time IS NOT NULL THEN
        TIMESTAMP_DIFF(
          SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', close_time),
          SAFE.PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S%Ez', entry_time),
          MINUTE
        )
    END
  ) AS avg_hold_minutes,
  APPROX_QUANTILES(CASE WHEN pl_pips > 0 THEN pl_pips END, 20)[OFFSET(10)] AS tp_p50,
  APPROX_QUANTILES(CASE WHEN pl_pips < 0 THEN pl_pips END, 20)[OFFSET(10)] AS sl_p50
FROM base
GROUP BY pocket, strategy, regime
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("lookback_days", "INT64", self.lookback_days),
            ]
        )
        try:
            job = self.bq_client.query(sql, job_config=job_config)
            return list(job.result())
        except gexc.NotFound:
            logging.warning("[BQ] trades table not found: %s", table_ref)
            return []
        except Exception as exc:  # noqa: BLE001
            logging.exception("[BQ] query failed: %s", exc)
            return []

    def _build_score(self, row: bigquery.table.Row) -> Optional[StrategyScore]:
        pocket = str(row.get("pocket") or "unknown").lower()
        strategy = str(row.get("strategy") or "unknown")
        regime = str(row.get("regime") or "all")
        trade_count = int(row.get("trade_count") or 0)
        if trade_count < max(1, self.min_trades):
            return None

        win = float(row.get("winning_trades") or 0.0)
        win_rate = win / trade_count if trade_count else 0.0
        gross_profit = float(row.get("gross_profit") or 0.0)
        gross_loss = float(row.get("gross_loss") or 0.0)
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf") if gross_profit > 0 else 0.0
        avg_pips = float(row.get("avg_pips") or 0.0)
        total_pips = float(row.get("total_pips") or 0.0)
        std_pips = float(row.get("std_pips") or 0.0)
        sharpe_like = total_pips / std_pips if std_pips > 1e-9 else 0.0
        tp_p50 = row.get("tp_p50")
        sl_p50 = row.get("sl_p50")
        lot_multiplier, score = self._score_metrics(
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_like=sharpe_like,
        )
        tp_pips, sl_pips = self._derive_sltp(tp_p50, sl_p50, avg_pips)
        return StrategyScore(
            strategy=strategy,
            pocket=pocket,
            regime=regime,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_like=sharpe_like,
            avg_pips=avg_pips,
            std_pips=std_pips,
            lot_multiplier=lot_multiplier,
            score=score,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )

    @staticmethod
    def _score_metrics(
        *,
        trade_count: int,
        win_rate: float,
        profit_factor: float,
        sharpe_like: float,
    ) -> Tuple[float, float]:
        # score は win, PF, 安定度の簡易合成
        pf_c = max(0.0, min(3.0, profit_factor if math.isfinite(profit_factor) else 3.0))
        win_c = max(0.0, min(1.0, win_rate))
        stab = max(0.0, min(2.0, abs(sharpe_like)))
        volume = 1.0 + min(0.4, math.log1p(trade_count) / 10.0)  # サンプルが多いほどわずかに加点
        score = (0.55 * pf_c + 0.35 * win_c * 2 + 0.1 * stab) * volume
        # ロット倍率は 0.7〜1.3 にクランプ（安全側）
        raw_mult = 1.0
        if pf_c > 1.1 and win_c > 0.52:
            raw_mult += (pf_c - 1.0) * 0.12 + (win_c - 0.52) * 0.8
        elif pf_c < 0.9 or win_c < 0.48:
            raw_mult -= (0.9 - min(pf_c, 0.9)) * 0.15 + (0.48 - min(win_c, 0.48)) * 0.6
        lot_multiplier = max(0.7, min(1.3, raw_mult))
        return lot_multiplier, score

    @staticmethod
    def _derive_sltp(tp_p50: Any, sl_p50: Any, avg_pips: float) -> Tuple[Optional[float], Optional[float]]:
        tp = None
        sl = None
        try:
            tp_val = float(tp_p50) if tp_p50 is not None else None
            sl_val = float(sl_p50) if sl_p50 is not None else None
        except (TypeError, ValueError):
            tp_val = None
            sl_val = None

        if tp_val is not None and math.isfinite(tp_val) and tp_val > 0:
            tp = max(2.0, min(30.0, tp_val))
        elif avg_pips > 0:
            tp = max(2.0, min(30.0, avg_pips * 1.5))

        if sl_val is not None and math.isfinite(sl_val) and sl_val < 0:
            sl = max(2.0, min(25.0, abs(sl_val)))
        elif avg_pips < 0:
            sl = max(2.0, min(25.0, abs(avg_pips) * 1.2))

        return tp, sl


def payload_summary(payload: Dict[str, Any]) -> str:
    scores = payload.get("strategy_scores", [])
    return json.dumps(
        {
            "count": len(scores),
            "updated_at": payload.get("updated_at"),
            "lookback_days": payload.get("lookback_days"),
        },
        ensure_ascii=False,
    )

