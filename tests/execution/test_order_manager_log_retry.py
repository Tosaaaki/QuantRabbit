from __future__ import annotations

import logging
import sqlite3

import execution.order_manager as order_manager


class _FakeConnection:
    def __init__(self, lock_failures: int) -> None:
        self._lock_failures = max(0, int(lock_failures))
        self.execute_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    def execute(self, _sql: str, _params: object = None) -> None:
        self.execute_calls += 1
        if self._lock_failures > 0:
            self._lock_failures -= 1
            raise sqlite3.OperationalError("database is locked")

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:  # pragma: no cover - compatibility hook
        return None


def _invoke_log_order() -> None:
    order_manager._log_order(
        pocket="scalp_fast",
        instrument="USD_JPY",
        side="buy",
        units=1000,
        sl_price=None,
        tp_price=None,
        client_order_id="cid-test",
        status="test_status",
        attempt=1,
        request_payload={"k": "v"},
        response_payload={"ok": True},
    )


def test_log_order_retries_locked_once_and_succeeds(monkeypatch, caplog) -> None:
    con = _FakeConnection(lock_failures=1)
    reset_calls: list[int] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_SLEEP_SEC", 0.01)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_BACKOFF", 2.0)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC", 0.05)
    monkeypatch.setattr(order_manager, "_orders_con", lambda: con)
    monkeypatch.setattr(order_manager, "_reset_orders_con", lambda: reset_calls.append(1))
    monkeypatch.setattr(order_manager, "_maybe_checkpoint_orders_db", lambda _con: None)
    monkeypatch.setattr(order_manager.time, "sleep", lambda sec: sleep_calls.append(sec))

    with caplog.at_level(logging.WARNING):
        _invoke_log_order()

    assert con.execute_calls == 2
    assert con.commit_calls == 1
    assert con.rollback_calls == 1
    assert len(reset_calls) == 1
    assert len(sleep_calls) == 1
    assert "[ORDER][LOG] failed to persist orders log" not in caplog.text


def test_log_order_emits_warning_after_retry_budget_exhausted(monkeypatch, caplog) -> None:
    con = _FakeConnection(lock_failures=8)
    reset_calls: list[int] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_SLEEP_SEC", 0.01)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_BACKOFF", 2.0)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC", 0.05)
    monkeypatch.setattr(order_manager, "_orders_con", lambda: con)
    monkeypatch.setattr(order_manager, "_reset_orders_con", lambda: reset_calls.append(1))
    monkeypatch.setattr(order_manager, "_maybe_checkpoint_orders_db", lambda _con: None)
    monkeypatch.setattr(order_manager.time, "sleep", lambda sec: sleep_calls.append(sec))

    with caplog.at_level(logging.WARNING):
        _invoke_log_order()

    assert con.execute_calls == 3
    assert con.commit_calls == 0
    assert con.rollback_calls == 2
    assert len(reset_calls) == 2
    assert len(sleep_calls) == 2
    assert "[ORDER][LOG] failed to persist orders log" in caplog.text
