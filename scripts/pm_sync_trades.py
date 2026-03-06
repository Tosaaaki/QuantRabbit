#!/usr/bin/env python3
"""Sync trades.db via local PositionManager service (report-only helper).

Calls:
  POST http://127.0.0.1:${POSITION_MANAGER_SERVICE_PORT:-8301}/position/sync_trades
  JSON {"max_fetch": 120}

Output is intentionally short:
  - ok trades=<n>
  - err <message>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sync trades.db via local PositionManager service")
    ap.add_argument("--max-fetch", type=int, default=120)
    ap.add_argument("--timeout", type=float, default=9.0)
    return ap.parse_args()


def _result_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return len(payload)
    return 0


def main() -> int:
    ns = _parse_args()
    port = _env_int("POSITION_MANAGER_SERVICE_PORT", 8301)
    max_fetch = max(1, int(ns.max_fetch))
    timeout_sec = max(0.5, float(ns.timeout))
    url = f"http://127.0.0.1:{port}/position/sync_trades"

    try:
        resp = requests.post(
            url,
            json={"max_fetch": max_fetch},
            timeout=timeout_sec,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"err request_failed:{exc}", file=sys.stderr)
        return 1

    try:
        body = resp.json()
    except Exception as exc:
        text = (resp.text or "").strip()
        snippet = text[:200].replace("\n", " ") if text else ""
        print(f"err non_json_response:{exc} body={snippet}", file=sys.stderr)
        return 2

    if isinstance(body, dict) and "ok" in body:
        if not bool(body.get("ok")):
            msg = str(body.get("error") or "service returned ok=false").strip()
            print(f"err {msg}", file=sys.stderr)
            return 3
        result = body.get("result")
        print(f"ok trades={_result_count(result)}")
        return 0

    # Backward/alternative response shapes
    print(f"ok trades={_result_count(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

