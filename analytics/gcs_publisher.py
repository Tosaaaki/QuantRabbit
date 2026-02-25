from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - fallback to CLI upload
    storage = None

from utils.secrets import get_secret
from utils.gcs_uploader import metadata_available, upload_json_via_metadata

_DEFAULT_OBJECT = "realtime/ui_state.json"
_DEFAULT_BUCKET_KEYS = ("ui_bucket_name", "GCS_BACKUP_BUCKET")
_UPLOAD_TIMEOUT_SEC = max(
    1.0, float(os.getenv("UI_SNAPSHOT_GCS_UPLOAD_TIMEOUT_SEC", "8.0"))
)


def _is_placeholder(value: str) -> bool:
    return "placeholder" in str(value).strip().lower()


def _resolve_bucket_name() -> str | None:
    for key in _DEFAULT_BUCKET_KEYS:
        try:
            value = get_secret(key)
        except KeyError:
            continue
        if not value or _is_placeholder(value):
            continue
        if key != "ui_bucket_name":
            logging.warning(
                "[GCS] ui_bucket_name が未設定のため %s にフォールバックします。", key
            )
        return str(value)
    return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GCSRealtimePublisher:
    """UI 向けに最新状態を GCS へ書き出す。"""

    def __init__(self, object_path: str | None = None) -> None:
        self._enabled = True
        if object_path:
            self._object_path = object_path
        else:
            try:
                configured = get_secret("ui_state_object_path")
            except KeyError:
                configured = None
            configured = str(configured).strip() if configured is not None else ""
            self._object_path = configured or _DEFAULT_OBJECT
        self._bucket_name: str | None = None
        self._use_cli = False
        self._use_metadata = False
        try:
            project_id = get_secret("gcp_project_id")
        except KeyError:
            project_id = None

        bucket_name = _resolve_bucket_name()
        if not bucket_name:
            logging.warning(
                "[GCS] ui_bucket_name が設定されていないため、リアルタイム出力は無効化されます。"
            )
            self._enabled = False
            self._client = None
            self._bucket = None
            return
        self._bucket_name = str(bucket_name)
        self._use_metadata = metadata_available()

        try:
            if storage is None:
                raise RuntimeError("google-cloud-storage not available")
            self._client = storage.Client(project=project_id) if project_id else storage.Client()
            self._bucket = self._client.bucket(bucket_name)
        except Exception as exc:  # noqa: BLE001
            logging.exception("[GCS] クライアント初期化に失敗しました: %s", exc)
            self._client = None
            self._bucket = None
            if shutil.which("gcloud") or shutil.which("gsutil"):
                self._use_cli = True
            elif not self._use_metadata:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and (
            self._bucket is not None or self._use_cli or self._use_metadata
        )

    def _upload_via_cli(self, payload: str) -> bool:
        if not self._bucket_name:
            return False
        target = f"gs://{self._bucket_name}/{self._object_path}"
        for cmd in (["gcloud", "storage", "cp", "-", target], ["gsutil", "cp", "-", target]):
            if not shutil.which(cmd[0]):
                continue
            try:
                proc = subprocess.run(
                    cmd,
                    input=payload,
                    text=True,
                    capture_output=True,
                    timeout=10.0,
                    check=False,
                )
            except Exception:
                continue
            if proc.returncode == 0:
                logging.info("[GCS] realtime snapshot uploaded via %s", cmd[0])
                return True
        return False

    def publish_snapshot(
        self,
        *,
        new_trades: Sequence[Mapping[str, Any]],
        recent_trades: Sequence[Mapping[str, Any]],
        open_positions: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        generated_at: str | None = None,
        snapshot_mode: str | None = None,
        snapshot_source: str | None = None,
    ) -> None:
        """UI が参照するスナップショットを GCS に保存する。"""
        if not self.enabled:
            return

        payload: dict[str, Any] = {
            "generated_at": generated_at or _utcnow_iso(),
            "new_trades": list(new_trades),
            "recent_trades": list(recent_trades),
            "open_positions": dict(open_positions),
            "metrics": dict(metrics) if metrics else {},
        }
        if snapshot_mode:
            payload["snapshot_mode"] = snapshot_mode
        if snapshot_source:
            payload["snapshot_source"] = snapshot_source

        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if self._bucket is not None:
            try:
                blob = self._bucket.blob(self._object_path)
                blob.cache_control = "no-cache"
                blob.upload_from_string(
                    serialized,
                    content_type="application/json",
                    timeout=_UPLOAD_TIMEOUT_SEC,
                )
                logging.info(
                    "[GCS] realtime snapshot uploaded: trades=%d recent=%d",
                    len(payload["new_trades"]),
                    len(payload["recent_trades"]),
                )
                return
            except Exception as exc:  # noqa: BLE001
                logging.warning("[GCS] realtime snapshot upload failed: %s", exc)
        if self._use_cli and self._upload_via_cli(serialized):
            return
        if self._use_metadata and self._bucket_name:
            if upload_json_via_metadata(
                self._bucket_name,
                self._object_path,
                serialized,
                cache_control="no-cache",
            ):
                logging.info("[GCS] realtime snapshot uploaded via metadata")
                return
