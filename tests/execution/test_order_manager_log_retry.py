from __future__ import annotations

import asyncio
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


class _LimitSnapshot:
    def __init__(self) -> None:
        self.nav = 1_000_000.0
        self.margin_used = 20_000.0
        self.margin_rate = 0.04


class _SideCapSnapshot:
    def __init__(self) -> None:
        # usage_total ~= 3.2%
        self.nav = 58_000.0
        self.margin_used = 1_860.0
        self.margin_rate = 0.04


def _invoke_log_order(*, fast_fail: bool = False) -> None:
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
        fast_fail=fast_fail,
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
    assert con.rollback_calls == 3
    assert len(reset_calls) == 3
    assert len(sleep_calls) == 2
    assert "[ORDER][LOG] failed to persist orders log" in caplog.text


def test_log_order_fast_fail_uses_short_retry_budget(monkeypatch, caplog) -> None:
    con = _FakeConnection(lock_failures=8)
    reset_calls: list[int] = []
    sleep_calls: list[float] = []

    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_SLEEP_SEC", 0.01)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_BACKOFF", 2.0)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC", 0.05)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_FAST_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_FAST_RETRY_SLEEP_SEC", 0.0)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_FAST_RETRY_BACKOFF", 1.0)
    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_FAST_RETRY_MAX_SLEEP_SEC", 0.0)
    monkeypatch.setattr(order_manager, "_orders_con", lambda: con)
    monkeypatch.setattr(order_manager, "_reset_orders_con", lambda: reset_calls.append(1))
    monkeypatch.setattr(order_manager, "_maybe_checkpoint_orders_db", lambda _con: None)
    monkeypatch.setattr(order_manager.time, "sleep", lambda sec: sleep_calls.append(sec))

    with caplog.at_level(logging.INFO):
        _invoke_log_order(fast_fail=True)

    assert con.execute_calls == 1
    assert con.commit_calls == 0
    assert con.rollback_calls == 1
    assert len(reset_calls) == 1
    assert len(sleep_calls) == 0
    assert "fast-fail dropped by lock" in caplog.text


def test_market_order_skips_preservice_db_log_when_service_mode(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict) -> str:
        return "TICKET-1"

    logged_statuses: list[str] = []

    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE", False)
    monkeypatch.setattr(order_manager, "_order_manager_service_enabled", lambda: True)
    monkeypatch.setattr(
        order_manager,
        "_order_manager_service_request_async",
        _fake_service_request_async,
    )
    monkeypatch.setattr(
        order_manager,
        "_probability_scaled_units",
        lambda *_args, **_kwargs: (500, None),
    )
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: logged_statuses.append(str(kwargs.get("status"))))
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)

    ticket = asyncio.run(
        order_manager.market_order(
            instrument="USD_JPY",
            units=1000,
            sl_price=None,
            tp_price=155.2,
            pocket="scalp_fast",
            client_order_id="cid-preservice-skip",
            strategy_tag="scalp_ping_5s_b_live",
            entry_thesis={"entry_probability": 0.95, "entry_units_intent": 1000},
            confidence=95,
        )
    )

    assert ticket == "TICKET-1"
    assert logged_statuses == []


def test_market_order_service_none_result_does_not_fallback_local(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict) -> None:
        return None

    monkeypatch.setattr(order_manager, "_ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE", False)
    monkeypatch.setattr(order_manager, "_order_manager_service_enabled", lambda: True)
    monkeypatch.setattr(
        order_manager,
        "_order_manager_service_request_async",
        _fake_service_request_async,
    )
    monkeypatch.setattr(
        order_manager,
        "_probability_scaled_units",
        lambda *_args, **_kwargs: (1000, None),
    )
    monkeypatch.setattr(
        order_manager,
        "_log_order",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local_fallback_called")),
    )
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)

    ticket = asyncio.run(
        order_manager.market_order(
            instrument="USD_JPY",
            units=1000,
            sl_price=None,
            tp_price=155.2,
            pocket="scalp_fast",
            client_order_id="cid-service-none",
            strategy_tag="scalp_ping_5s_c_live",
            entry_thesis={"entry_probability": 0.40, "entry_units_intent": 1000},
            confidence=40,
        )
    )

    assert ticket is None


def test_limit_order_service_none_result_does_not_fallback_local(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict) -> None:
        return None

    monkeypatch.setattr(order_manager, "_order_manager_service_enabled", lambda: True)
    monkeypatch.setattr(
        order_manager,
        "_order_manager_service_request_async",
        _fake_service_request_async,
    )
    monkeypatch.setattr(
        order_manager,
        "_probability_scaled_units",
        lambda *_args, **_kwargs: (1000, None),
    )
    monkeypatch.setattr(
        order_manager,
        "_log_order",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local_fallback_called")),
    )
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_args, **_kwargs: None)

    trade_id, order_id = asyncio.run(
        order_manager.limit_order(
            instrument="USD_JPY",
            units=1000,
            price=156.000,
            sl_price=155.900,
            tp_price=None,
            pocket="scalp_fast",
            client_order_id="cid-limit-service-none",
            entry_thesis={"entry_probability": 0.95, "entry_units_intent": 1000},
            confidence=95,
        )
    )

    assert trade_id is None
    assert order_id is None


def test_limit_order_quote_retry_runs_even_when_submit_attempts_is_one(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict):
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    submit_calls = {"count": 0}
    statuses: list[str] = []

    def _fake_api_request(endpoint) -> None:
        submit_calls["count"] += 1
        if submit_calls["count"] == 1:
            endpoint.response = {
                "orderRejectTransaction": {
                    "rejectReason": "OFF_QUOTES",
                    "errorCode": "OFF_QUOTES",
                    "errorMessage": "off quotes",
                }
            }
            return
        endpoint.response = {
            "orderCreateTransaction": {"id": "OID-1001"},
        }

    monkeypatch.setenv("ORDER_SUBMIT_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(order_manager, "_ORDER_QUOTE_RETRY_MAX_RETRIES", 1)
    monkeypatch.setattr(order_manager.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request_async)
    monkeypatch.setattr(order_manager, "_probability_scaled_units", lambda *_a, **_k: (1000, None))
    monkeypatch.setattr(order_manager, "_entry_sl_disabled_for_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_disable_hard_stop_by_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_soft_tp_mode", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_entry_hard_stop_pips", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_entry_loss_cap_jpy", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_apply_default_entry_thesis_tfs", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_regime", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_flags", lambda thesis: thesis)
    monkeypatch.setattr(
        order_manager,
        "_augment_entry_thesis_policy_generation",
        lambda thesis, reduce_only=False: thesis,
    )
    monkeypatch.setattr(order_manager, "_manual_margin_pressure_details", lambda **_k: None)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)
    monkeypatch.setattr(order_manager, "_is_passive_price", lambda **_k: True)
    monkeypatch.setattr(
        order_manager,
        "_fetch_quote_with_retry",
        lambda *_a, **_k: {"bid": 150.000, "ask": 150.010, "mid": 150.005, "spread_pips": 1.0},
    )
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: statuses.append(str(kwargs.get("status"))))
    monkeypatch.setattr(order_manager.api, "request", _fake_api_request)
    monkeypatch.setattr("utils.oanda_account.get_account_snapshot", lambda cache_ttl_sec=1.0: _LimitSnapshot())
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda *args, **kwargs: (0.0, 0.0))

    trade_id, order_id = asyncio.run(
        order_manager.limit_order(
            instrument="USD_JPY",
            units=1000,
            price=150.000,
            sl_price=149.950,
            tp_price=150.120,
            pocket="scalp",
            client_order_id="cid-limit-quote-retry",
            entry_thesis={"entry_probability": 0.95, "entry_units_intent": 1000},
            confidence=95,
        )
    )

    assert submit_calls["count"] == 2
    assert trade_id is None
    assert order_id == "OID-1001"
    assert "quote_retry" in statuses


def test_limit_order_allows_net_reducing_under_side_cap(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict):
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    submit_calls = {"count": 0}
    statuses: list[str] = []

    def _fake_api_request(endpoint) -> None:
        submit_calls["count"] += 1
        endpoint.response = {
            "orderCreateTransaction": {"id": "OID-SIDECAP-1"},
        }

    monkeypatch.setenv("MARGIN_SIDE_CAP_ENABLED", "1")
    monkeypatch.setenv("MAX_MARGIN_USAGE", "0.92")
    monkeypatch.setenv("MAX_MARGIN_USAGE_HARD", "0.96")
    monkeypatch.setattr(order_manager, "_ORDER_QUOTE_RETRY_MAX_RETRIES", 0)
    monkeypatch.setattr(order_manager.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request_async)
    monkeypatch.setattr(order_manager, "_probability_scaled_units", lambda *_a, **_k: (-180, None))
    monkeypatch.setattr(order_manager, "_entry_sl_disabled_for_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_disable_hard_stop_by_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_soft_tp_mode", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_entry_hard_stop_pips", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_entry_loss_cap_jpy", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_apply_default_entry_thesis_tfs", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_regime", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_flags", lambda thesis: thesis)
    monkeypatch.setattr(
        order_manager,
        "_augment_entry_thesis_policy_generation",
        lambda thesis, reduce_only=False: thesis,
    )
    monkeypatch.setattr(order_manager, "_manual_margin_pressure_details", lambda **_k: None)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)
    monkeypatch.setattr(order_manager, "_is_passive_price", lambda **_k: True)
    monkeypatch.setattr(
        order_manager,
        "_fetch_quote_with_retry",
        lambda *_a, **_k: {"bid": 155.000, "ask": 155.010, "mid": 155.005, "spread_pips": 1.0},
    )
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: statuses.append(str(kwargs.get("status"))))
    monkeypatch.setattr(order_manager.api, "request", _fake_api_request)
    monkeypatch.setattr(
        "utils.oanda_account.get_account_snapshot",
        lambda cache_ttl_sec=1.0: _SideCapSnapshot(),
    )
    # Long/short both large (hedged) but net is small:
    # sell -180 reduces net exposure while side_cap would appear high.
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda *args, **kwargs: (8900.0, 8700.0))

    trade_id, order_id = asyncio.run(
        order_manager.limit_order(
            instrument="USD_JPY",
            units=-180,
            price=155.000,
            sl_price=155.200,
            tp_price=154.900,
            pocket="scalp",
            client_order_id="cid-limit-net-reducing-side-cap",
            entry_thesis={
                "strategy_tag": "scalp_ping_5s_b_live",
                "entry_probability": 0.95,
                "entry_units_intent": 180,
            },
            meta={"entry_price": 155.000},
            confidence=95,
        )
    )

    assert submit_calls["count"] == 1
    assert trade_id is None
    assert order_id == "OID-SIDECAP-1"
    assert "margin_usage_projected_cap" not in statuses
    assert "margin_usage_exceeds_cap" not in statuses


def test_entry_probability_value_returns_none_when_candidates_are_non_numeric() -> None:
    prob = order_manager._entry_probability_value(
        confidence=None,
        entry_thesis={"entry_probability": "nan-text", "confidence": object()},
    )
    assert prob is None


def test_market_order_entry_intent_guard_rejects_missing_probability(monkeypatch) -> None:
    async def _unexpected_service_call(_path: str, _payload: dict) -> None:
        raise AssertionError("service should not be called when entry intent guard rejects")

    statuses: list[str] = []

    monkeypatch.setattr(order_manager, "_ORDER_MANAGER_REQUIRE_ENTRY_INTENT_FIELDS", True)
    monkeypatch.setattr(order_manager, "_ORDER_MANAGER_REQUIRE_ENTRY_INTENT_PROBABILITY", True)
    monkeypatch.setattr(order_manager, "_ORDER_MANAGER_REQUIRE_STRATEGY_TAG_FOR_ENTRY", True)
    monkeypatch.setattr(order_manager, "_should_persist_preservice_order_log", lambda: True)
    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _unexpected_service_call)
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: statuses.append(str(kwargs.get("status"))))
    monkeypatch.setattr(order_manager, "log_metric", lambda *_a, **_k: None)

    ticket = asyncio.run(
        order_manager.market_order(
            instrument="USD_JPY",
            units=1200,
            sl_price=None,
            tp_price=155.2,
            pocket="scalp_fast",
            client_order_id="cid-entry-intent-guard",
            strategy_tag="scalp_ping_5s_b_live",
            entry_thesis={"entry_units_intent": 1200},
            confidence=None,
        )
    )

    assert ticket is None
    assert "entry_intent_guard_reject" in statuses


def test_limit_order_retries_with_rotated_client_id_on_duplicate_reject(monkeypatch) -> None:
    async def _fake_service_request_async(_path: str, _payload: dict):
        return order_manager._ORDER_MANAGER_SERVICE_UNHANDLED

    submit_client_ids: list[str] = []
    statuses: list[str] = []

    def _fake_api_request(endpoint) -> None:
        submit_client_ids.append(str(endpoint.data["order"]["clientExtensions"].get("id") or ""))
        if len(submit_client_ids) == 1:
            endpoint.response = {
                "orderRejectTransaction": {
                    "rejectReason": "CLIENT_TRADE_ID_ALREADY_EXISTS",
                    "errorCode": "CLIENT_TRADE_ID_ALREADY_EXISTS",
                    "errorMessage": "duplicate client id",
                }
            }
            return
        endpoint.response = {
            "orderCreateTransaction": {"id": "OID-DUP-2"},
        }

    monkeypatch.setenv("ORDER_SUBMIT_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(order_manager, "_ORDER_QUOTE_RETRY_MAX_RETRIES", 0)
    monkeypatch.setattr(order_manager, "_order_manager_service_request_async", _fake_service_request_async)
    monkeypatch.setattr(order_manager, "_probability_scaled_units", lambda *_a, **_k: (1000, None))
    monkeypatch.setattr(order_manager, "_entry_sl_disabled_for_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_disable_hard_stop_by_strategy", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_soft_tp_mode", lambda *_a, **_k: False)
    monkeypatch.setattr(order_manager, "_entry_hard_stop_pips", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_entry_loss_cap_jpy", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(order_manager, "_apply_default_entry_thesis_tfs", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_regime", lambda thesis, _pocket: thesis)
    monkeypatch.setattr(order_manager, "_augment_entry_thesis_flags", lambda thesis: thesis)
    monkeypatch.setattr(
        order_manager,
        "_augment_entry_thesis_policy_generation",
        lambda thesis, reduce_only=False: thesis,
    )
    monkeypatch.setattr(order_manager, "_manual_margin_pressure_details", lambda **_k: None)
    monkeypatch.setattr(order_manager, "is_market_open", lambda: True)
    monkeypatch.setattr(order_manager, "_is_passive_price", lambda **_k: True)
    monkeypatch.setattr(
        order_manager,
        "_fetch_quote_with_retry",
        lambda *_a, **_k: {"bid": 150.000, "ask": 150.010, "mid": 150.005, "spread_pips": 1.0},
    )
    monkeypatch.setattr(order_manager, "_latest_filled_trade_id_by_client_id", lambda _cid: None)
    monkeypatch.setattr(
        order_manager,
        "_rotate_client_order_id",
        lambda _cid, *, pocket, strategy_tag: f"qr-rotated-{pocket}-{strategy_tag}",
    )
    monkeypatch.setattr(order_manager, "_console_order_log", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "log_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(order_manager, "_log_order", lambda **kwargs: statuses.append(str(kwargs.get("status"))))
    monkeypatch.setattr(order_manager.api, "request", _fake_api_request)
    monkeypatch.setattr("utils.oanda_account.get_account_snapshot", lambda cache_ttl_sec=1.0: _LimitSnapshot())
    monkeypatch.setattr("utils.oanda_account.get_position_summary", lambda *args, **kwargs: (0.0, 0.0))

    trade_id, order_id = asyncio.run(
        order_manager.limit_order(
            instrument="USD_JPY",
            units=1000,
            price=150.000,
            sl_price=149.950,
            tp_price=150.120,
            pocket="scalp",
            client_order_id="cid-limit-dup-retry",
            entry_thesis={
                "strategy_tag": "scalp_ping_5s_b_live",
                "entry_probability": 0.88,
                "entry_units_intent": 1000,
            },
            confidence=88,
        )
    )

    assert trade_id is None
    assert order_id == "OID-DUP-2"
    assert len(submit_client_ids) == 2
    assert submit_client_ids[0] == "cid-limit-dup-retry"
    assert submit_client_ids[1] != submit_client_ids[0]
    assert submit_client_ids[1].startswith("qr-rotated-scalp-")
    assert "duplicate_client_id_retry" in statuses
