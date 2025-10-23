from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from google.cloud import storage

from utils.secrets import get_secret

_DEFAULT_OBJECT = "realtime/ui_state.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GCSRealtimePublisher:
    """UI 向けに最新状態を GCS へ書き出す。"""

    def __init__(self, object_path: str | None = None) -> None:
        self._enabled = True
        self._object_path = object_path or _DEFAULT_OBJECT
        try:
            project_id = get_secret("gcp_project_id")
        except KeyError:
            project_id = None

        try:
            bucket_name = get_secret("ui_bucket_name")
        except KeyError:
            logging.warning(
                "[GCS] ui_bucket_name が設定されていないため、リアルタイム出力は無効化されます。"
            )
            self._enabled = False
            self._client = None
            self._bucket = None
            return

        try:
            self._client = storage.Client(project=project_id) if project_id else storage.Client()
            self._bucket = self._client.bucket(bucket_name)
        except Exception as exc:  # noqa: BLE001
            logging.exception("[GCS] クライアント初期化に失敗しました: %s", exc)
            self._enabled = False
            self._client = None
            self._bucket = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._bucket is not None

    def publish_snapshot(
        self,
        *,
        new_trades: Sequence[Mapping[str, Any]],
        recent_trades: Sequence[Mapping[str, Any]],
        open_positions: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        generated_at: str | None = None,
    ) -> None:
        """UI が参照するスナップショットを GCS に保存する。"""
        if not self.enabled:
            return

        payload = {
            "generated_at": generated_at or _utcnow_iso(),
            "new_trades": list(new_trades),
            "recent_trades": list(recent_trades),
            "open_positions": dict(open_positions),
            "metrics": dict(metrics) if metrics else {},
        }

        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        blob = self._bucket.blob(self._object_path)
        blob.cache_control = "no-cache"
        blob.upload_from_string(serialized, content_type="application/json")
        logging.info(
            "[GCS] realtime snapshot uploaded: trades=%d recent=%d",
            len(payload["new_trades"]),
            len(payload["recent_trades"]),
        )
