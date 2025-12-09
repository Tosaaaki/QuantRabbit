from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    firestore = None

from utils.secrets import get_secret


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(default)).strip().lower()
    return val in {"1", "true", "yes", "on"}


@dataclass
class _Cache:
    payload: Dict[str, Any]
    fetched_ts: float


class FirestoreStrategyClient:
    """TTL キャッシュ付きで Firestore の strategy_scores/current を読むリーダー。

    フォールバック: 無効/読めない場合は乗数 1.0、SL/TP 変更なし。
    """

    def __init__(
        self,
        collection: str | None = None,
        document: str | None = None,
        ttl_sec: int | None = None,
        project: Optional[str] = None,
        enable: Optional[bool] = None,
    ) -> None:
        self.enabled = True if enable is None else bool(enable)
        self.collection = collection or os.getenv("FIRESTORE_COLLECTION", "strategy_scores")
        self.document = document or os.getenv("FIRESTORE_DOCUMENT", "current")
        self.ttl_sec = ttl_sec or int(os.getenv("FIRESTORE_STRATEGY_TTL_SEC", "60"))
        self.project = project or os.getenv("FIRESTORE_PROJECT") or os.getenv("BQ_PROJECT") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            try:
                self.project = get_secret("gcp_project_id")
            except Exception:
                self.project = None
        self._cache: Optional[_Cache] = None
        self._client = None
        if self.enabled and firestore is not None:
            try:
                self._client = firestore.Client(project=self.project)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[FS] init failed: %s", exc)
                self._client = None
        else:
            if self.enabled:
                logging.warning("[FS] google-cloud-firestore is not installed; disabling client.")
            self.enabled = False

    def refresh(self, *, force: bool = False) -> None:
        if not self.enabled or not self._client:
            return
        now = time.time()
        if not force and self._cache and now - self._cache.fetched_ts < self.ttl_sec:
            return
        try:
            doc_ref = self._client.collection(self.collection).document(self.document)
            snap = doc_ref.get()
            if snap.exists:
                data = snap.to_dict() or {}
                self._cache = _Cache(payload=data, fetched_ts=now)
                logging.info(
                    "[FS] fetched strategy_scores %s/%s (count=%s, updated_at=%s)",
                    self.collection,
                    self.document,
                    len(data.get("strategy_scores", []) or []),
                    data.get("updated_at"),
                )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[FS] refresh failed: %s", exc)

    def get_multiplier(self, strategy: str, pocket: str) -> float:
        if not self.enabled:
            return 1.0
        if not self._cache:
            self.refresh(force=True)
        scores = (self._cache.payload.get("strategy_scores") if self._cache else None) or []
        pocket_low = str(pocket).lower()
        strategy_key = str(strategy)
        best = None
        for item in scores:
            try:
                if item.get("pocket") != pocket_low:
                    continue
                if item.get("strategy") != strategy_key:
                    continue
                best = item
                break
            except Exception:
                continue
        if not best:
            return 1.0
        try:
            mult = float(best.get("lot_multiplier", 1.0))
            if not (0.2 < mult < 5.0):
                return 1.0
            return mult
        except Exception:
            return 1.0

    def get_sltp(self, strategy: str, pocket: str, regime: str | None = None) -> Tuple[Optional[float], Optional[float]]:
        if not self.enabled:
            return None, None
        if not self._cache:
            self.refresh(force=True)
        scores = (self._cache.payload.get("strategy_scores") if self._cache else None) or []
        pocket_low = str(pocket).lower()
        strategy_key = str(strategy)
        regime_key = (regime or "all").lower()
        best = None
        for item in scores:
            try:
                if item.get("pocket") != pocket_low:
                    continue
                if item.get("strategy") != strategy_key:
                    continue
                # regime はオプション。完全一致を優先し、なければ全件のうち最初を使う。
                if best is None:
                    best = item
                if item.get("regime", "").lower() == regime_key:
                    best = item
                    break
            except Exception:
                continue
        if not best:
            return None, None
        tp = best.get("tp_pips")
        sl = best.get("sl_pips")
        try:
            tp_val = float(tp) if tp is not None else None
        except Exception:
            tp_val = None
        try:
            sl_val = float(sl) if sl is not None else None
        except Exception:
            sl_val = None
        return tp_val, sl_val


def firestore_strategy_enabled() -> bool:
    return _env_bool("FIRESTORE_STRATEGY_ENABLE", default=False)
