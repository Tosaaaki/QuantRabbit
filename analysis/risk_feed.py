from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from google.cloud import pubsub_v1


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


_STATE_LOCK = threading.Lock()
_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}
_STATE_PATH = Path(os.getenv("RISK_MODEL_STATE", "logs/risk_scores.json"))
_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
_CURRENT_FUTURE: Optional[pubsub_v1.subscriber.futures.StreamingPullFuture] = None
_CURRENT_SUBSCRIBER: Optional[pubsub_v1.SubscriberClient] = None
_STOP_EVENT = threading.Event()


def _load_initial_state() -> None:
    if not _STATE_PATH.exists():
        return
    try:
        data = json.loads(_STATE_PATH.read_text())
        entries = data.get("entries", [])
        _apply_entries(entries, {"ts": data.get("ts"), "model": data.get("model"), "source": data.get("source")})
        logging.info("[RISK_FEED] loaded %s cached risk scores", len(entries))
    except Exception as exc:
        logging.warning("[RISK_FEED] failed to load state: %s", exc)


def _persist_state(meta: Dict[str, object]) -> None:
    snapshot = snapshot_entries()
    payload = {
        "ts": meta.get("ts") or datetime.now(timezone.utc).isoformat(),
        "model": meta.get("model"),
        "source": meta.get("source"),
        "entries": snapshot,
    }
    try:
        _STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:
        logging.warning("[RISK_FEED] failed to persist state: %s", exc)


def _apply_entries(entries, meta: Dict[str, object]) -> int:
    changed = 0
    if not isinstance(entries, list):
        return 0
    with _STATE_LOCK:
        for raw in entries:
            try:
                pocket = str(raw.get("pocket") or "").lower()
                strategy = str(raw.get("strategy") or "")
                multiplier = raw.get("multiplier")
            except AttributeError:
                continue
            if not pocket or not strategy or multiplier is None:
                continue
            key = (pocket, strategy)
            record = {
                "multiplier": _to_float(multiplier) or 1.0,
                "score": _to_float(raw.get("score")) or 0.0,
                "predicted_pf": _to_float(raw.get("predicted_pf")),
                "pf": _to_float(raw.get("pf")),
                "win_rate": _to_float(raw.get("win_rate")),
                "trades": int(raw.get("trades") or 0),
                "trade_date": str(raw.get("trade_date") or ""),
                "source": meta.get("source") or "risk-model",
                "model": meta.get("model"),
                "updated_at": meta.get("ts") or datetime.now(timezone.utc).isoformat(),
            }
            previous = _CACHE.get(key)
            if previous == record:
                continue
            _CACHE[key] = record
            changed += 1
    if changed:
        _persist_state(meta)
    return changed


def _on_message(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        payload = json.loads(message.data.decode("utf-8"))
    except Exception as exc:
        logging.warning("[RISK_FEED] invalid payload: %s", exc)
        message.ack()
        return
    if payload.get("type") != "risk_scores":
        message.ack()
        return
    entries = payload.get("entries", [])
    changed = _apply_entries(entries, payload)
    if changed:
        logging.info("[RISK_FEED] updated %s entries (model=%s)", changed, payload.get("model"))
    message.ack()


def _subscriber_runner(subscription: str) -> None:
    global _CURRENT_FUTURE, _CURRENT_SUBSCRIBER
    client = pubsub_v1.SubscriberClient()
    try:
        future = client.subscribe(subscription, callback=_on_message)
        with _STATE_LOCK:
            _CURRENT_FUTURE = future
            _CURRENT_SUBSCRIBER = client
        logging.info("[RISK_FEED] listening on %s", subscription)
        try:
            future.result()
        except Exception as exc:
            if _STOP_EVENT.is_set():
                logging.info("[RISK_FEED] subscription cancelled")
            else:
                logging.warning("[RISK_FEED] subscriber stopped: %s", exc)
            raise
    finally:
        with _STATE_LOCK:
            _CURRENT_FUTURE = None
            _CURRENT_SUBSCRIBER = None
        try:
            client.close()
        except Exception:
            pass


def _cancel_subscription() -> None:
    with _STATE_LOCK:
        future = _CURRENT_FUTURE
    if future:
        future.cancel()


async def risk_feed_loop() -> None:
    subscription = (
        os.getenv("RISK_MODEL_SUBSCRIPTION")
        or os.getenv("RISK_PUBSUB_SUBSCRIPTION")
        or os.getenv("RISK_FEED_SUBSCRIPTION")
    )
    if not subscription:
        logging.info("[RISK_FEED] disabled (no subscription configured)")
        while True:
            await asyncio.sleep(300)
    _load_initial_state()
    loop = asyncio.get_running_loop()
    while True:
        _STOP_EVENT.clear()
        try:
            await loop.run_in_executor(None, _subscriber_runner, subscription)
        except asyncio.CancelledError:
            _STOP_EVENT.set()
            _cancel_subscription()
            raise
        except Exception:
            if not _STOP_EVENT.is_set():
                logging.info("[RISK_FEED] restarting subscriber in 10 seconds")
                await asyncio.sleep(10)


def get_multiplier(pocket: str, strategy: str) -> Optional[float]:
    key = (str(pocket or "").lower(), str(strategy or ""))
    with _STATE_LOCK:
        record = _CACHE.get(key)
        if not record:
            return None
        mult = record.get("multiplier")
    return _to_float(mult)


def get_entry(pocket: str, strategy: str) -> Optional[Dict[str, object]]:
    key = (str(pocket or "").lower(), str(strategy or ""))
    with _STATE_LOCK:
        record = _CACHE.get(key)
        if not record:
            return None
        return dict(record)


def snapshot_entries() -> list[Dict[str, object]]:
    with _STATE_LOCK:
        items = []
        for (pocket, strategy), record in _CACHE.items():
            item = {"pocket": pocket, "strategy": strategy}
            item.update(record)
            items.append(item)
    items.sort(key=lambda item: (item["pocket"], item["strategy"]))
    return items


try:
    _load_initial_state()
except Exception as exc:  # pragma: no cover
    logging.warning("[RISK_FEED] initial state load failed: %s", exc)
