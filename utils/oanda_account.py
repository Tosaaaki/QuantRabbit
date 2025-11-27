from __future__ import annotations

import dataclasses
import requests
from typing import Optional

from utils.secrets import get_secret


@dataclasses.dataclass
class AccountSnapshot:
    nav: float
    balance: float
    margin_available: float
    margin_used: float
    margin_rate: float
    unrealized_pl: float
    free_margin_ratio: Optional[float]
    health_buffer: Optional[float]


_LAST_SNAPSHOT: AccountSnapshot | None = None
_LAST_SNAPSHOT_TS: float | None = None


def get_account_snapshot(timeout: float = 7.0, *, cache_ttl_sec: float = 600.0) -> AccountSnapshot:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except KeyError:
        practice = False
    base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"

    headers = {"Authorization": f"Bearer {token}"}
    global _LAST_SNAPSHOT, _LAST_SNAPSHOT_TS

    try:
        resp = requests.get(
            f"{base}/v3/accounts/{account}/summary",
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        acc = resp.json().get("account", {})
    except Exception as exc:
        if _LAST_SNAPSHOT and _LAST_SNAPSHOT_TS is not None:
            import time

            age = time.time() - _LAST_SNAPSHOT_TS
            if age <= cache_ttl_sec:
                return _LAST_SNAPSHOT
        raise

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(acc.get(key, default))
        except (TypeError, ValueError):
            return default

    nav = _f("NAV", 0.0)
    balance = _f("balance", nav)
    margin_available = _f("marginAvailable", 0.0)
    margin_used = _f("marginUsed", 0.0)
    margin_rate = _f("marginRate", 0.0)
    unrealized = _f("unrealizedPL", 0.0)
    margin_closeout = _f("marginCloseoutPercent", 0.0)

    free_ratio: Optional[float]
    if nav > 0:
        free_ratio = margin_available / nav
    else:
        free_ratio = None

    health_buffer: Optional[float]
    if margin_closeout > 0:
        health_buffer = max(0.0, 1.0 - margin_closeout)
    elif margin_used > 0:
        health_buffer = max(0.0, margin_available / (margin_used + 1e-9))
    else:
        health_buffer = free_ratio

    snapshot = AccountSnapshot(
        nav=nav,
        balance=balance,
        margin_available=margin_available,
        margin_used=margin_used,
        margin_rate=margin_rate,
        unrealized_pl=unrealized,
        free_margin_ratio=free_ratio,
        health_buffer=health_buffer,
    )
    try:
        import time

        _LAST_SNAPSHOT = snapshot
        _LAST_SNAPSHOT_TS = time.time()
    except Exception:
        pass
    return snapshot
