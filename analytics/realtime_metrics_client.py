"""Helper utilities to read realtime KPI snapshots and recommendations.

The trading loop can leverage this module to make data-driven decisions without
having to embed BigQuery specific logic at the call site. The client caches the
latest rows fetched from BigQuery to minimise latency and gracefully falls back
when BigQuery is unavailable (e.g. during local development).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from google.api_core import exceptions as gexc
from google.cloud import bigquery


DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
METRICS_TABLE = os.getenv("BQ_REALTIME_METRICS_TABLE", "realtime_metrics")
RECO_TABLE = os.getenv("BQ_RECOMMENDATION_TABLE", "strategy_recommendations")
TTL_SECONDS = int(os.getenv("REALTIME_METRICS_TTL", "240"))


@dataclass
class StrategyHealth:
    pocket: str
    strategy: str
    win_rate: float
    profit_factor: float
    max_drawdown_pips: float
    losing_streak: int
    total_trades: int
    confidence_scale: float = 1.0
    allowed: bool = True
    reason: Optional[str] = None


class RealtimeMetricsClient:
    def __init__(self, project: Optional[str] = None, dataset: str = DEFAULT_DATASET):
        self._project = project
        self._dataset = dataset
        self._client: Optional[bigquery.Client] = None
        self._last_fetch = datetime.min.replace(tzinfo=timezone.utc)
        self._cache: Dict[str, StrategyHealth] = {}

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            self._client = (
                bigquery.Client(project=self._project)
                if self._project
                else bigquery.Client()
            )
        return self._client

    def refresh(self) -> None:
        now = datetime.now(timezone.utc)
        if (now - self._last_fetch).total_seconds() < TTL_SECONDS:
            return

        try:
            metrics = self._fetch_metrics()
            recos = self._fetch_recommendations()
        except gexc.NotFound:
            logging.warning("[REALTIME] metrics tables not ready.")
            return
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("[REALTIME] refresh failed: %s", exc)
            return

        merged: Dict[str, StrategyHealth] = {}
        for row in metrics:
            key = f"{row['pocket']}::{row['strategy']}"
            merged[key] = StrategyHealth(
                pocket=row["pocket"],
                strategy=row["strategy"],
                win_rate=row.get("win_rate") or 0.0,
                profit_factor=row.get("profit_factor") or 0.0,
                max_drawdown_pips=row.get("max_drawdown_pips") or 0.0,
                losing_streak=int(row.get("losing_streak") or 0),
                total_trades=int(row.get("total_trades") or 0),
            )

        # Apply suggestion overrides (confidence scaling, ban lists, etc.)
        for reco in recos:
            key = f"{reco['pocket']}::{reco['strategy']}"
            health = merged.get(key)
            if not health:
                health = StrategyHealth(
                    pocket=reco["pocket"],
                    strategy=reco["strategy"],
                    win_rate=0.0,
                    profit_factor=0.0,
                    max_drawdown_pips=0.0,
                    losing_streak=0,
                    total_trades=0,
                )
                merged[key] = health

            scale = reco.get("confidence_scale")
            if scale is not None:
                health.confidence_scale = float(scale)
            action = (reco.get("action") or "").lower()
            if action in {"halt", "suspend"}:
                health.allowed = False
                health.reason = reco.get("reason") or "auto_suspended"
            elif action in {"caution", "decrease"}:
                health.confidence_scale = min(health.confidence_scale, 0.5)
                health.reason = reco.get("reason") or "confidence_decreased"

        self._cache = merged
        self._last_fetch = now

    def _fetch_metrics(self):
        query = f"""
        SELECT * EXCEPT(row_num)
        FROM (
          SELECT *,
            ROW_NUMBER() OVER (PARTITION BY pocket, strategy ORDER BY generated_at DESC) AS row_num
          FROM `{self.client.project}.{self._dataset}.{METRICS_TABLE}`
        )
        WHERE row_num = 1
        """
        return list(self.client.query(query).result())

    def _fetch_recommendations(self):
        table = f"{self.client.project}.{self._dataset}.{RECO_TABLE}"
        try:
            job = self.client.query(
                f"""
                SELECT * EXCEPT(row_num) FROM (
                  SELECT *, ROW_NUMBER() OVER (PARTITION BY pocket, strategy ORDER BY generated_at DESC) AS row_num
                  FROM `{table}`
                ) WHERE row_num = 1
                """
            )
            return list(job.result())
        except gexc.NotFound:
            return []

    def evaluate(self, strategy: str, pocket: str) -> StrategyHealth:
        key = f"{pocket}::{strategy}"
        health = self._cache.get(key)
        if health:
            return health
        # fallback: default neutral health
        return StrategyHealth(
            pocket=pocket,
            strategy=strategy,
            win_rate=1.0,
            profit_factor=1.0,
            max_drawdown_pips=0.0,
            losing_streak=0,
            total_trades=0,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()


class ConfidencePolicy:
    """Utility to compute confidence scaling based on KPI thresholds."""

    def __init__(
        self,
        min_trades: int = int(os.getenv("CONF_POLICY_MIN_TRADES", "5")),
        win_floor: float = float(os.getenv("CONF_POLICY_WIN_FLOOR", "0.45")),
        win_boost: float = float(os.getenv("CONF_POLICY_WIN_BOOST", "0.6")),
        pf_floor: float = float(os.getenv("CONF_POLICY_PF_FLOOR", "0.9")),
        pf_boost: float = float(os.getenv("CONF_POLICY_PF_BOOST", "1.1")),
        dd_cap: float = float(os.getenv("CONF_POLICY_MAX_DD", "30")),
        streak_cap: int = int(os.getenv("CONF_POLICY_MAX_STREAK", "4")),
    ) -> None:
        self.min_trades = min_trades
        self.win_floor = win_floor
        self.win_boost = win_boost
        self.pf_floor = pf_floor
        self.pf_boost = pf_boost
        self.dd_cap = dd_cap
        self.streak_cap = streak_cap

    def apply(self, health: StrategyHealth) -> StrategyHealth:
        if health.total_trades < self.min_trades:
            return health

        if health.max_drawdown_pips >= self.dd_cap or health.losing_streak >= self.streak_cap:
            health.allowed = False
            health.reason = "risk_guard_drawdown"
            return health

        # penalise
        if health.win_rate <= self.win_floor or health.profit_factor <= self.pf_floor:
            health.confidence_scale = min(health.confidence_scale, 0.4)
            health.reason = health.reason or "low_performance"
            return health

        if health.win_rate >= self.win_boost and health.profit_factor >= self.pf_boost:
            health.confidence_scale = max(health.confidence_scale, 1.1)

        return health

    def reset(self) -> None:
        """Maintain compatibility with callers that expect a reset() hook."""
        return None
