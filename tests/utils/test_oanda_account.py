from __future__ import annotations

import json
import pathlib
import sys
import time

import requests

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.oanda_account as oanda_account


def _fake_secret(name: str) -> str:
    values = {
        "oanda_token": "token",
        "oanda_account_id": "account",
        "oanda_practice": "false",
    }
    return values[name]


def test_get_account_snapshot_state_uses_stale_disk_cache_on_http_503(monkeypatch, tmp_path):
    now = time.time()
    monkeypatch.setattr(oanda_account, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(oanda_account, "_SHARED_CACHE_ENABLED", True)
    monkeypatch.setattr(oanda_account, "_LAST_SNAPSHOT", None)
    monkeypatch.setattr(oanda_account, "_LAST_SNAPSHOT_TS", None)
    monkeypatch.setattr(oanda_account, "_LAST_SNAPSHOT_ENV", None)
    monkeypatch.setattr(oanda_account, "get_secret", _fake_secret)

    cache_path, _ = oanda_account._account_cache_paths("live")
    cache_path.write_text(
        json.dumps(
            {
                "ts": now - 2.0,
                "data": {
                    "nav": 100000.0,
                    "balance": 100000.0,
                    "margin_available": 60000.0,
                    "margin_used": 25000.0,
                    "margin_rate": 0.04,
                    "unrealized_pl": 0.0,
                    "free_margin_ratio": 0.6,
                    "health_buffer": 0.55,
                },
            }
        ),
        encoding="utf-8",
    )

    def _fail(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 503
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(oanda_account.requests, "get", _fail)
    state = oanda_account.get_account_snapshot_state(cache_ttl_sec=1.0, allow_stale_sec=5.0)

    assert state.stale is True
    assert state.source == "disk_cache"
    assert state.error_kind == "http_503"
    assert state.age_sec >= 2.0
    assert state.snapshot.margin_available == 60000.0


def test_get_position_summary_uses_stale_disk_cache_on_request_error(monkeypatch, tmp_path):
    now = time.time()
    monkeypatch.setattr(oanda_account, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(oanda_account, "_SHARED_CACHE_ENABLED", True)
    oanda_account._POS_STATE.clear()
    monkeypatch.setattr(oanda_account, "get_secret", _fake_secret)

    cache_path, _ = oanda_account._pos_cache_paths("live", "USD_JPY")
    cache_path.write_text(
        json.dumps(
            {
                "ts": now - 2.0,
                "instrument": "USD_JPY",
                "long_units": 1200.0,
                "short_units": 300.0,
            }
        ),
        encoding="utf-8",
    )

    def _fail(*_args, **_kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(oanda_account.requests, "get", _fail)

    long_units, short_units = oanda_account.get_position_summary("USD_JPY", cache_ttl_sec=1.0)

    assert long_units == 1200.0
    assert short_units == 300.0


def test_side_free_margin_ratio_keeps_global_ratio_when_positions_unavailable(monkeypatch):
    snap = oanda_account.AccountSnapshot(
        nav=100000.0,
        balance=100000.0,
        margin_available=60000.0,
        margin_used=25000.0,
        margin_rate=0.04,
        unrealized_pl=0.0,
        free_margin_ratio=0.6,
        health_buffer=0.55,
    )
    monkeypatch.setattr(oanda_account, "get_position_summary", lambda: (0.0, 0.0))

    side_ratio = oanda_account._side_free_margin_ratio(snap)

    assert side_ratio is None


def test_side_free_margin_ratio_does_not_expand_one_sided_exposure(monkeypatch):
    snap = oanda_account.AccountSnapshot(
        nav=100000.0,
        balance=100000.0,
        margin_available=75000.0,
        margin_used=25000.0,
        margin_rate=0.04,
        unrealized_pl=0.0,
        free_margin_ratio=0.75,
        health_buffer=0.55,
    )
    monkeypatch.setattr(oanda_account, "get_position_summary", lambda: (10000.0, 0.0))

    side_ratio = oanda_account._side_free_margin_ratio(snap)

    assert side_ratio == 0.75


def test_apply_side_free_margin_ratio_never_raises_global_ratio(monkeypatch):
    snap = oanda_account.AccountSnapshot(
        nav=100000.0,
        balance=100000.0,
        margin_available=3000.0,
        margin_used=97000.0,
        margin_rate=0.04,
        unrealized_pl=0.0,
        free_margin_ratio=0.03,
        health_buffer=0.03,
    )
    monkeypatch.setattr(oanda_account, "_side_free_margin_ratio", lambda _snapshot: 1.0)

    applied = oanda_account._apply_side_free_margin_ratio(snap)

    assert applied is snap
    assert applied.free_margin_ratio == 0.03
