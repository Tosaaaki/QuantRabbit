from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from google.cloud import storage

from utils.secrets import get_secret


@dataclass
class _Cache:
    data: Dict[str, Dict[str, float]]
    generated_at: str
    fetched_ts: float


class InsightClient:
    """Fetches lot_insights.json from GCS and exposes pocket/side multipliers.

    JSON format:
      {
        "generated_at": "...",
        "insights": [
           {"pocket":"macro","side":"LONG","risk_multiplier":1.1, ...},
           {"pocket":"macro","side":"ALL","risk_multiplier":1.0, ...},
        ]
      }
    """

    def __init__(self, object_path: str = "analytics/lot_insights.json", ttl_sec: int = 600):
        self._ttl = ttl_sec
        self._object_path = object_path
        self._bucket = self._resolve_bucket()
        self._client = storage.Client() if self._bucket else None
        self._cache: Optional[_Cache] = None

    @staticmethod
    def _resolve_bucket() -> Optional[str]:
        try:
            return get_secret("analytics_bucket_name")
        except Exception:
            try:
                return get_secret("ui_bucket_name")
            except Exception:
                return None

    def refresh(self, force: bool = False) -> None:
        if not self._bucket or not self._client:
            return
        now = time.time()
        if not force and self._cache and now - self._cache.fetched_ts < self._ttl:
            return
        try:
            blob = self._client.bucket(self._bucket).blob(self._object_path)
            raw = blob.download_as_text()
            js = json.loads(raw)
            mapping: Dict[str, Dict[str, float]] = {}
            for ins in js.get("insights", []):
                pocket = str(ins.get("pocket", "")).strip() or "unknown"
                side = str(ins.get("side", "ALL")).upper()
                mult = float(ins.get("risk_multiplier", 1.0))
                key = f"{pocket}:{side}"
                mapping[key] = max(0.5, min(1.6, mult))
            gen = js.get("generated_at") or ""
            self._cache = _Cache(mapping, gen, now)
            logging.info("[INSIGHT] loaded %d multipliers (gen=%s)", len(mapping), gen)
        except Exception as exc:
            logging.warning("[INSIGHT] refresh failed: %s", exc)

    def get_multiplier(self, pocket: str, side: str) -> float:
        if not self._cache:
            self.refresh(force=True)
        data = self._cache.data if self._cache else {}
        side_up = side.upper()
        val = data.get(f"{pocket}:{side_up}")
        if val is not None:
            return float(val)
        val = data.get(f"{pocket}:ALL")
        return float(val) if val is not None else 1.0

