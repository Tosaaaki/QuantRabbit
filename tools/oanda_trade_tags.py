#!/usr/bin/env python3
"""Helpers for recovering worker trade tags from OANDA transaction truth.

Some OANDA trade endpoints do not reliably echo the original tag/comment on the
trade object itself. The opening ORDER_FILL transaction does carry that data in
`tradeOpened.clientExtensions`, so recover from transactions when needed.
"""
from __future__ import annotations

from typing import Callable


ApiGetter = Callable[[str], dict]


def get_tag(payload: dict) -> str:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = payload.get(key, {}) or {}
        tag = ext.get("tag")
        if tag:
            return str(tag)
    return ""


def get_extensions(payload: dict) -> dict:
    for key in ("clientExtensions", "tradeClientExtensions"):
        ext = payload.get(key, {}) or {}
        if ext:
            return dict(ext)
    return {}


def _extensions_from_transaction(tx: dict, trade_id: str) -> dict:
    if not tx:
        return {}
    trade_opened = tx.get("tradeOpened") or {}
    if str(trade_opened.get("tradeID", "")) != str(trade_id):
        return {}
    return get_extensions(trade_opened)


def resolve_trade_extensions(trade_id: str, api_get: ApiGetter) -> dict:
    trade_id = str(trade_id or "").strip()
    if not trade_id:
        return {}

    try:
        tx = (api_get(f"/v3/accounts/{{account_id}}/transactions/{trade_id}") or {}).get("transaction", {})
        ext = _extensions_from_transaction(tx, trade_id)
        if ext:
            return ext
    except Exception:
        pass

    try:
        center = int(trade_id)
    except ValueError:
        return {}

    start = max(1, center - 3)
    end = center + 3
    try:
        txs = (api_get(f"/v3/accounts/{{account_id}}/transactions/idrange?from={start}&to={end}") or {}).get("transactions", [])
    except Exception:
        return {}
    for tx in txs:
        ext = _extensions_from_transaction(tx, trade_id)
        if ext:
            return ext
    return {}


def attach_trade_extensions(trade: dict, api_get: ApiGetter) -> dict:
    if get_tag(trade):
        return trade
    trade_id = str(trade.get("id", "")).strip()
    if not trade_id:
        return trade
    ext = resolve_trade_extensions(trade_id, api_get)
    if not ext:
        return trade
    enriched = dict(trade)
    if not enriched.get("tradeClientExtensions"):
        enriched["tradeClientExtensions"] = dict(ext)
    if not enriched.get("clientExtensions"):
        enriched["clientExtensions"] = dict(ext)
    return enriched


def enrich_open_trades(trades: list[dict], api_get: ApiGetter) -> list[dict]:
    cache: dict[str, dict] = {}
    enriched: list[dict] = []
    for trade in trades:
        if get_tag(trade):
            enriched.append(trade)
            continue
        trade_id = str(trade.get("id", "")).strip()
        if not trade_id:
            enriched.append(trade)
            continue
        ext = cache.get(trade_id)
        if ext is None:
            ext = resolve_trade_extensions(trade_id, api_get)
            cache[trade_id] = ext
        if not ext:
            enriched.append(trade)
            continue
        patched = dict(trade)
        if not patched.get("tradeClientExtensions"):
            patched["tradeClientExtensions"] = dict(ext)
        if not patched.get("clientExtensions"):
            patched["clientExtensions"] = dict(ext)
        enriched.append(patched)
    return enriched
