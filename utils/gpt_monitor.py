from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Mapping, Optional

from utils.metrics_logger import log_metric


def _sanitize(val: object) -> object:
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    return str(val)


class GPTCallTracker:
    __slots__ = ("_source", "_extra", "_tags", "_start", "_status")

    def __init__(self, source: str, extra_tags: Optional[Mapping[str, object]] = None) -> None:
        self._source = source
        self._extra = {k: _sanitize(v) for k, v in (extra_tags or {}).items() if v is not None}
        self._tags: dict[str, object] = {}
        self._start = time.perf_counter()
        self._status: Optional[str] = None

    def add_tag(self, key: str, value: object) -> None:
        sanitized = _sanitize(value)
        if sanitized is not None:
            self._tags[key] = sanitized

    def set_model(self, model: Optional[str]) -> None:
        if model:
            self._tags["model"] = str(model)

    def set_reason(self, reason: Optional[str]) -> None:
        if reason:
            self._tags["reason"] = str(reason)[:80]

    def mark_status(self, status: str) -> None:
        self._status = status

    def mark_failure(self, exc: BaseException) -> None:
        self._status = "error"
        self._tags["error"] = type(exc).__name__
        message = str(exc)
        if message:
            self._tags["error_msg"] = message[:120]

    def finalize(self) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000.0
        status = self._status or "success"
        tags = {"source": self._source, "status": status, **self._extra, **self._tags}
        log_metric("gpt_call_latency_ms", latency_ms, tags=tags)
        if status != "success":
            log_metric("gpt_call_errors", 1.0, tags=tags)


@contextmanager
def track_gpt_call(
    source: str,
    *,
    extra_tags: Optional[Mapping[str, object]] = None,
):
    tracker = GPTCallTracker(source, extra_tags)
    try:
        yield tracker
    except BaseException as exc:
        tracker.mark_failure(exc)
        tracker.finalize()
        raise
    else:
        if tracker._status is None:  # type: ignore[attr-defined]
            tracker.mark_status("success")
        tracker.finalize()


def log_gpt_fallback(
    source: str,
    reason: str,
    *,
    extra_tags: Optional[Mapping[str, object]] = None,
) -> None:
    tags = {"source": source, "reason": reason}
    if extra_tags:
        tags.update({k: _sanitize(v) for k, v in extra_tags.items() if v is not None})
    log_metric("gpt_call_fallback", 1.0, tags=tags)
