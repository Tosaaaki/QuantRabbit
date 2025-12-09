from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google.cloud import storage

from utils.secrets import get_secret


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Cache:
    payload: Dict[str, Any]
    fetched_ts: float


class LevelMapClient:
    """GCS 上の level_map.json を TTL キャッシュ付きで読み込む簡易クライアント。

    level_map.json フォーマット（generate_level_map.py の出力）:
    {
      "updated_at": "...",
      "bucket_pips": 5.0,
      "levels": [
        {"bucket": 154.2, "hit_count": 120, "mean_ret_5": ..., "p_up_5": ...},
        ...
      ]
    }
    """

    def __init__(
        self,
        object_path: str = "analytics/level_map.json",
        ttl_sec: int = 300,
    ) -> None:
        self._ttl = ttl_sec
        self._object_path = object_path
        self._bucket = self._resolve_bucket()
        self._client = storage.Client() if self._bucket else None
        self._cache: Optional[_Cache] = None
        self.enabled = _env_bool("LEVEL_MAP_ENABLE", False)
        if not self._bucket:
            self.enabled = False
        if not self.enabled:
            logging.info("[LEVEL_MAP] disabled (env flag or bucket missing)")

    @staticmethod
    def _resolve_bucket() -> Optional[str]:
        for key in ("analytics_bucket_name", "ui_bucket_name"):
            try:
                return get_secret(key)
            except Exception:
                continue
        return os.getenv("LEVEL_MAP_BUCKET")

    def refresh(self, force: bool = False) -> None:
        if not self.enabled or not self._client or not self._bucket:
            return
        now = time.time()
        if not force and self._cache and now - self._cache.fetched_ts < self._ttl:
            return
        try:
            blob = self._client.bucket(self._bucket).blob(self._object_path)
            raw = blob.download_as_text()
            payload = json.loads(raw)
            levels = payload.get("levels", [])
            if not isinstance(levels, list):
                levels = []
            # normalize buckets into a dict
            bucket_map = {}
            for item in levels:
                try:
                    b = float(item.get("bucket"))
                except Exception:
                    continue
                bucket_map[b] = item
            payload["bucket_map"] = bucket_map
            self._cache = _Cache(payload=payload, fetched_ts=now)
            logging.info(
                "[LEVEL_MAP] loaded %d buckets (gen=%s)",
                len(bucket_map),
                payload.get("updated_at"),
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[LEVEL_MAP] refresh failed: %s", exc)

    def nearest(self, price: float) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        if not self._cache:
            self.refresh(force=True)
        if not self._cache:
            return None
        bucket_map = self._cache.payload.get("bucket_map", {}) or {}
        if not bucket_map:
            return None
        # find nearest bucket by absolute difference
        nearest_b = None
        nearest_dist = float("inf")
        for b in bucket_map.keys():
            dist = abs(b - price)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_b = b
        if nearest_b is None:
            return None
        return bucket_map.get(nearest_b)

