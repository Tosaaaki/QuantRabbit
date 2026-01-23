from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google.api_core import exceptions as gexc
from google.cloud import bigquery

try:  # optional
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None

try:  # optional
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover
    firestore = None

from utils.gcs_uploader import metadata_available, upload_json_via_metadata
from utils.secrets import get_secret
from analytics.policy_diff import policy_hash

DEFAULT_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
DEFAULT_POLICY_TABLE = os.getenv("BQ_POLICY_TABLE", "policy_history")
DEFAULT_GCS_PREFIX = os.getenv("POLICY_GCS_PREFIX", "policy")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_gcs_bucket() -> Optional[str]:
    for key in ("analytics_bucket_name", "ui_bucket_name"):
        try:
            return get_secret(key)
        except Exception:
            continue
    return os.getenv("GCS_BACKUP_BUCKET") or None


class PolicyLedger:
    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        dataset_id: str = DEFAULT_DATASET,
        table_id: str = DEFAULT_POLICY_TABLE,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = DEFAULT_GCS_PREFIX,
        enable_bq: bool = True,
        enable_gcs: bool = True,
        enable_firestore: bool = False,
        firestore_collection: str = "policy_diffs",
        firestore_document: str = "latest",
    ) -> None:
        self.project_id = project_id or os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.enable_bq = enable_bq
        self.enable_gcs = enable_gcs
        self.enable_firestore = enable_firestore
        self.firestore_collection = firestore_collection
        self.firestore_document = firestore_document

        self._bq_client = None
        self._bq_ready = False
        if self.enable_bq:
            try:
                self._bq_client = (
                    bigquery.Client(project=self.project_id)
                    if self.project_id
                    else bigquery.Client()
                )
                self._ensure_bq_table()
                self._bq_ready = True
            except Exception as exc:
                logging.warning("[POLICY_LEDGER] BigQuery disabled: %s", exc)
                self._bq_client = None

        self._gcs_bucket = gcs_bucket or _resolve_gcs_bucket()
        self._gcs_client = None
        self._gcs_bucket_obj = None
        self._gcs_use_cli = False
        self._gcs_use_metadata = False
        if self.enable_gcs and self._gcs_bucket:
            self._gcs_use_metadata = metadata_available()
            try:
                if storage is None:
                    raise RuntimeError("google-cloud-storage not available")
                self._gcs_client = storage.Client(project=self.project_id) if self.project_id else storage.Client()
                self._gcs_bucket_obj = self._gcs_client.bucket(self._gcs_bucket)
            except Exception as exc:
                logging.warning("[POLICY_LEDGER] GCS client init failed: %s", exc)
                self._gcs_client = None
                self._gcs_bucket_obj = None
                if shutil.which("gcloud") or shutil.which("gsutil"):
                    self._gcs_use_cli = True

        self._fs_client = None
        if self.enable_firestore:
            if firestore is None:
                logging.warning("[POLICY_LEDGER] Firestore unavailable; disabling.")
                self.enable_firestore = False
            else:
                try:
                    self._fs_client = firestore.Client(project=self.project_id)
                except Exception as exc:
                    logging.warning("[POLICY_LEDGER] Firestore init failed: %s", exc)
                    self._fs_client = None
                    self.enable_firestore = False

    def _ensure_bq_table(self) -> None:
        if not self._bq_client:
            return
        dataset_ref = bigquery.DatasetReference(self._bq_client.project, self.dataset_id)
        try:
            self._bq_client.get_dataset(dataset_ref)
        except gexc.NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = os.getenv("BQ_LOCATION", "US")
            self._bq_client.create_dataset(dataset, exists_ok=True)

        table_ref = dataset_ref.table(self.table_id)
        schema = [
            bigquery.SchemaField("policy_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("status", "STRING"),
            bigquery.SchemaField("policy_hash", "STRING"),
            bigquery.SchemaField("payload_json", "STRING"),
            bigquery.SchemaField("summary_json", "STRING"),
        ]
        try:
            self._bq_client.get_table(table_ref)
        except gexc.NotFound:
            self._bq_client.create_table(bigquery.Table(table_ref, schema=schema))

    def record(
        self,
        payload: Dict[str, Any],
        *,
        status: str = "generated",
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        policy_id = str(payload.get("policy_id") or f"policy-{int(datetime.now().timestamp())}")
        generated_at = payload.get("generated_at") or _utc_now_iso()
        source = payload.get("source") or "unknown"
        payload_text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        summary_text = (
            json.dumps(summary, ensure_ascii=True, separators=(",", ":")) if summary else None
        )
        phash = policy_hash(payload)

        if self._bq_ready and self._bq_client:
            row = {
                "policy_id": policy_id,
                "generated_at": generated_at,
                "source": source,
                "status": status,
                "policy_hash": phash,
                "payload_json": payload_text,
                "summary_json": summary_text,
            }
            table_ref = self._bq_client.dataset(self.dataset_id).table(self.table_id)
            try:
                errors = self._bq_client.insert_rows_json(table_ref, [row])
                if errors:
                    raise RuntimeError(errors)
            except Exception as exc:
                logging.warning("[POLICY_LEDGER] BQ insert failed: %s", exc)

        if self.enable_gcs and self._gcs_bucket:
            self._write_gcs(payload_text, policy_id=policy_id)

        if self.enable_firestore and self._fs_client:
            try:
                doc_ref = self._fs_client.collection(self.firestore_collection).document(self.firestore_document)
                doc_ref.set(payload)
            except Exception as exc:
                logging.warning("[POLICY_LEDGER] Firestore write failed: %s", exc)

    def _write_gcs(self, payload_text: str, *, policy_id: str) -> None:
        if not self._gcs_bucket:
            return
        object_latest = f"{self.gcs_prefix}/latest.json"
        object_history = f"{self.gcs_prefix}/history/{policy_id}.json"
        if self._gcs_bucket_obj is not None:
            try:
                for obj in (object_latest, object_history):
                    blob = self._gcs_bucket_obj.blob(obj)
                    blob.cache_control = "no-cache"
                    blob.upload_from_string(payload_text, content_type="application/json")
                return
            except Exception as exc:
                logging.warning("[POLICY_LEDGER] GCS upload failed: %s", exc)
        if self._gcs_use_cli:
            for obj in (object_latest, object_history):
                target = f"gs://{self._gcs_bucket}/{obj}"
                for cmd in (["gcloud", "storage", "cp", "-", target], ["gsutil", "cp", "-", target]):
                    if not shutil.which(cmd[0]):
                        continue
                    try:
                        proc = subprocess.run(
                            cmd,
                            input=payload_text,
                            text=True,
                            capture_output=True,
                            timeout=10.0,
                            check=False,
                        )
                    except Exception:
                        continue
                    if proc.returncode == 0:
                        break
        if self._gcs_use_metadata:
            upload_json_via_metadata(self._gcs_bucket, object_latest, payload_text, cache_control="no-cache")
            upload_json_via_metadata(self._gcs_bucket, object_history, payload_text, cache_control="no-cache")

