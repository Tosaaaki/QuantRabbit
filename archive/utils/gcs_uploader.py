from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Optional

_METADATA_HOST = os.environ.get("GCE_METADATA_HOST", "metadata.google.internal")
_METADATA_BASE = f"http://{_METADATA_HOST}/computeMetadata/v1"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def _metadata_get(path: str, timeout: float) -> Optional[str]:
    url = f"{_METADATA_BASE}/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers=_METADATA_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                return None
            return resp.read().decode("utf-8")
    except Exception:
        return None


def metadata_available(timeout: float = 0.2) -> bool:
    return _metadata_get("instance/id", timeout) is not None


def fetch_access_token(timeout: float = 0.2) -> Optional[str]:
    payload = _metadata_get("instance/service-accounts/default/token", timeout)
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        return None
    return token


def upload_bytes_via_metadata(
    bucket: str,
    object_path: str,
    payload: bytes,
    *,
    content_type: str = "application/octet-stream",
    cache_control: str | None = None,
    timeout: float = 10.0,
) -> bool:
    token = fetch_access_token()
    if not token:
        return False
    name = urllib.parse.quote(object_path, safe="/")
    url = (
        "https://storage.googleapis.com/upload/storage/v1/b/"
        f"{bucket}/o?uploadType=media&name={name}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "Content-Length": str(len(payload)),
    }
    if cache_control:
        headers["Cache-Control"] = cache_control
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if 200 <= status < 300:
                return True
            logging.warning("[GCS] metadata upload failed status=%s", status)
    except Exception as exc:  # noqa: BLE001
        logging.debug("[GCS] metadata upload error: %s", exc)
    return False


def upload_json_via_metadata(
    bucket: str,
    object_path: str,
    payload: str,
    *,
    cache_control: str | None = None,
    timeout: float = 10.0,
) -> bool:
    return upload_bytes_via_metadata(
        bucket,
        object_path,
        payload.encode("utf-8"),
        content_type="application/json",
        cache_control=cache_control,
        timeout=timeout,
    )
