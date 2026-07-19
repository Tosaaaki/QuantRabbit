import json

import pytest

from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError, _sha


@pytest.fixture()
def broker(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "t0")
    return b


def _replay_coordinate(phase="O"):
    return {
        "mode": "replay",
        "epoch": 1_735_689_600,
        "phase": phase,
        "granularity": "M1",
        "intrabar": "OHLC",
    }


def test_market_fill_at_real_ask_and_bid(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=5, sl_pips=None)
    pos = broker.positions[tid]
    assert pos.entry_price == 150.02  # ask
    assert pos.tp_price == pytest.approx(150.07)
    sid = broker.market_order("USD_JPY", "SHORT", 5_000)
    assert broker.positions[sid].entry_price == 150.00  # bid


def test_market_order_resolves_attached_stop_on_the_fill_quote(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=200_000.0)
    stamp = "2026-01-01T00:00:00+00:00#O"
    b.on_quote("USD_JPY", 100.00, 102.00, stamp)

    trade_id = b.market_order("USD_JPY", "LONG", 100, sl_pips=10)

    assert trade_id not in b.positions
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    owned = [row for row in records if row["payload"].get("trade_id") == trade_id]
    assert [row["event"] for row in owned] == ["FILL_MARKET", "EXIT_SL"]
    assert owned[1]["payload"]["price"] == 100.00
    assert owned[1]["payload"]["quote"]["ts"] == stamp


def test_no_quote_refuses_fill(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(VirtualBrokerError, match="no live quote"):
        b.market_order("USD_JPY", "LONG", 1000)


def test_tp_fills_only_when_quote_touches(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=5)
    events = broker.on_quote("USD_JPY", 150.06, 150.08, "t1")  # bid below TP
    assert not events
    events = broker.on_quote("USD_JPY", 150.07, 150.09, "t2")  # bid touches TP
    assert events[0]["event"] == "EXIT_TP"
    assert tid not in broker.positions
    assert broker.balance_jpy == pytest.approx(200_000 + 0.05 * 10_000)


def test_sl_first_when_both_touch_same_quote(broker):
    broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=2, sl_pips=3)
    # a quote where bid is below SL (gap through both is impossible on one
    # quote; SL-first rule applies when SL is touched)
    events = broker.on_quote("USD_JPY", 149.99, 150.01, "t1")
    assert events[0]["event"] == "EXIT_SL"


def test_sl_gap_fills_at_worse_price(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000, sl_pips=3)
    events = broker.on_quote("USD_JPY", 149.90, 149.92, "t1")  # gap through SL
    assert events[0]["event"] == "EXIT_SL"
    assert events[0]["price"] == 149.90  # the worse real bid, not the SL level


def test_limit_fill_at_level_or_better_never_synthesized(broker):
    oid = broker.limit_order("USD_JPY", "LONG", 10_000, price=149.95, tp_pips=10)
    assert not broker.on_quote("USD_JPY", 149.96, 149.98, "t1")
    events = broker.on_quote("USD_JPY", 149.92, 149.94, "t2")  # ask below limit
    assert events[0]["event"] == "FILL_LIMIT"
    assert events[0]["price"] == 149.94  # better real ask, not the level
    assert events[0]["entry"] == 149.94
    assert events[0]["tp"] == 150.04
    assert events[0]["sl"] is None
    assert events[0]["order_kind"] == "LIMIT"
    assert oid not in broker.orders


@pytest.mark.parametrize(
    ("kind", "side", "price", "expected_entry", "order_event"),
    [
        ("LIMIT", "LONG", 150.05, 150.02, "ORDER_LIMIT"),
        ("LIMIT", "SHORT", 149.95, 150.00, "ORDER_LIMIT"),
        ("STOP", "LONG", 150.01, 150.02, "ORDER_STOP"),
        ("STOP", "SHORT", 150.01, 150.00, "ORDER_STOP"),
    ],
)
def test_marketable_entry_order_fills_on_submission_quote(
    tmp_path, kind, side, price, expected_entry, order_event
):
    b = VirtualBroker(
        ledger_path=tmp_path / f"{kind}-{side}.jsonl",
        balance_jpy=2_000_000.0,
    )
    stamp = "2026-01-01T00:00:00+00:00#O"
    b.on_quote("USD_JPY", 150.00, 150.02, stamp)

    submit = b.limit_order if kind == "LIMIT" else b.stop_order
    order_id = submit("USD_JPY", side, 1_000, price=price)

    assert order_id not in b.orders
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [order_event, "FILL_LIMIT"]
    fill = records[1]["payload"]
    assert fill["order_id"] == order_id
    assert fill["order_kind"] == kind
    assert fill["entry"] == expected_entry
    assert fill["quote"] == {"bid": 150.00, "ask": 150.02, "ts": stamp}
    assert fill["trade_id"] in b.positions


@pytest.mark.parametrize(
    ("kind", "side", "price", "order_event"),
    [
        ("LIMIT", "LONG", 149.95, "ORDER_LIMIT"),
        ("LIMIT", "SHORT", 150.05, "ORDER_LIMIT"),
        ("STOP", "LONG", 150.10, "ORDER_STOP"),
        ("STOP", "SHORT", 149.90, "ORDER_STOP"),
    ],
)
def test_nonmarketable_entry_order_remains_resting(
    tmp_path, kind, side, price, order_event
):
    b = VirtualBroker(
        ledger_path=tmp_path / f"{kind}-{side}.jsonl",
        balance_jpy=2_000_000.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")

    submit = b.limit_order if kind == "LIMIT" else b.stop_order
    order_id = submit("USD_JPY", side, 1_000, price=price)

    assert order_id in b.orders
    assert not b.positions
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [order_event]


def test_stale_bar_gate_blocks_new_risk_but_permits_risk_reduction(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")
    trade_id = b.market_order("USD_JPY", "LONG", 1_000)
    order_id = b.limit_order("USD_JPY", "LONG", 1_000, price=149.90)

    with b.suspend_new_risk():
        with pytest.raises(VirtualBrokerError, match="new risk is suspended"):
            b.market_order("USD_JPY", "LONG", 1_000)
        with pytest.raises(VirtualBrokerError, match="new risk is suspended"):
            b.limit_order("USD_JPY", "LONG", 1_000, price=149.80)
        with pytest.raises(VirtualBrokerError, match="new risk is suspended"):
            b.stop_order("USD_JPY", "LONG", 1_000, price=150.10)
        b.cancel_order(order_id)
        b.close_trade(trade_id)

    assert not b.orders
    assert not b.positions


def test_marketable_entry_order_rejects_atomically_on_submission_quote(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=200_000.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")

    order_id = b.limit_order("USD_JPY", "LONG", 1_000_000, price=150.05)

    assert order_id not in b.orders
    assert not b.positions
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "ORDER_LIMIT",
        "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
    ]
    assert records[1]["payload"]["order_id"] == order_id


def test_submission_fill_resolves_attached_exit_on_same_quote(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
    )
    stamp = "2026-01-01T00:00:00+00:00#O"
    b.on_quote("USD_JPY", 100.00, 102.00, stamp)

    order_id = b.limit_order("USD_JPY", "LONG", 100, price=102.00, sl_pips=10)

    assert order_id not in b.orders
    assert not b.positions
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "ORDER_LIMIT",
        "FILL_LIMIT",
        "EXIT_SL",
    ]
    assert records[1]["payload"]["quote"]["ts"] == stamp
    assert records[2]["payload"]["quote"]["ts"] == stamp


def test_set_exit_resolves_already_crossed_stop_on_current_quote(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
    )
    stamp = "2026-01-01T00:00:00+00:00#O"
    b.on_quote("USD_JPY", 150.00, 150.02, stamp)
    trade_id = b.market_order("USD_JPY", "LONG", 1_000)

    b.set_exit(trade_id, sl_price=150.01)

    assert trade_id not in b.positions
    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "FILL_MARKET",
        "SET_EXIT",
        "EXIT_SL",
    ]
    assert records[-1]["payload"]["quote"]["ts"] == stamp
    assert records[-1]["payload"]["price"] == 150.00


def test_hedge_netting_margin(broker):
    broker.market_order("USD_JPY", "LONG", 10_000)
    acct_one = broker.account()
    broker.market_order("USD_JPY", "SHORT", 10_000)
    acct_hedged = broker.account()
    # opposite position adds no margin (max of sides unchanged)
    assert acct_hedged["margin_used_jpy"] == pytest.approx(
        acct_one["margin_used_jpy"], rel=1e-6
    )


def test_margin_closeout_liquidates_everything(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=200_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "t0")
    b.market_order("USD_JPY", "LONG", 30_000)  # ~90% usage at 25x
    events = b.on_quote("USD_JPY", 145.50, 145.52, "t1")  # huge adverse move
    assert any(e["event"] == "MARGIN_CLOSEOUT" for e in events)
    assert not b.positions


def test_eurusd_pl_converts_via_usdjpy(broker):
    broker.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    broker.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00")
    tid = broker.market_order("EUR_USD", "LONG", 10_000, tp_pips=10)
    events = broker.on_quote("EUR_USD", 1.1012, 1.1014, "2026-01-01T00:01:00+00:00")
    assert events[0]["event"] == "EXIT_TP"
    # 10 pips * 10k units = 10 USD -> ~1500 JPY at USDJPY mid 150.01
    assert broker.balance_jpy - 200_000 == pytest.approx(
        0.001 * 10_000 * 150.01, rel=1e-3
    )


def test_partial_close_and_manual_close(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000)
    broker.on_quote("USD_JPY", 150.10, 150.12, "t1")
    pl = broker.close_trade(tid, units=4_000)
    assert pl == pytest.approx((150.10 - 150.02) * 4_000)
    assert broker.positions[tid].units == 6_000
    broker.close_trade(tid)
    assert tid not in broker.positions


def test_ledger_chain_integrity(broker, tmp_path):
    broker.market_order("USD_JPY", "LONG", 1_000)
    broker.on_quote("USD_JPY", 150.05, 150.07, "t1")
    prev = "0" * 64
    for line in broker.ledger_path.read_text().splitlines():
        rec = json.loads(line)
        assert rec["prev_sha"] == prev
        prev = rec["sha"]


def test_invalid_quote_refused(broker):
    with pytest.raises(VirtualBrokerError, match="invalid quote"):
        broker.on_quote("USD_JPY", 150.05, 150.01, "t1")  # ask < bid


def test_snapshot_restore_roundtrip(broker, tmp_path):
    broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=5)
    broker.limit_order("USD_JPY", "SHORT", 5_000, price=150.50)
    snap = broker.snapshot()
    bound_ledger = tmp_path / "l2.jsonl"
    bound_ledger.write_bytes(broker.ledger_path.read_bytes())
    b2 = VirtualBroker(ledger_path=bound_ledger)
    b2.restore(snap)
    b2.on_quote("USD_JPY", 150.00, 150.02, "t0")
    assert b2.account()["open_positions"] == 1
    assert b2.account()["resting_orders"] == 1
    assert b2.balance_jpy == broker.balance_jpy


def test_snapshot_requires_exact_schema_and_bound_ledger(broker, tmp_path):
    broker.market_order("USD_JPY", "LONG", 1_000)
    snapshot = broker.snapshot()

    legacy = dict(snapshot)
    legacy["schema"] = "QR_VIRTUAL_BROKER_SNAPSHOT_V1"
    with pytest.raises(VirtualBrokerError, match="schema mismatch"):
        broker.restore(legacy)

    missing_field = dict(snapshot)
    missing_field.pop("feed_cursor")
    with pytest.raises(VirtualBrokerError, match="schema mismatch"):
        broker.restore(missing_field)

    unbound = VirtualBroker(ledger_path=tmp_path / "unbound.jsonl")
    with pytest.raises(VirtualBrokerError, match="ledger tip"):
        unbound.restore(snapshot)


def test_snapshot_sequence_cannot_reuse_existing_identity(broker):
    broker.market_order("USD_JPY", "LONG", 1_000)
    snapshot = broker.snapshot()
    snapshot["seq"] = 0
    original_sequence = broker._seq

    with pytest.raises(VirtualBrokerError, match="reuse an existing id"):
        broker.restore(snapshot)
    assert broker._seq == original_sequence


def test_stop_order_triggers_only_on_breakout(broker):
    oid = broker.stop_order("USD_JPY", "LONG", 10_000, price=150.10, tp_pips=5)
    assert not broker.on_quote("USD_JPY", 150.05, 150.07, "t1")  # below trigger
    events = broker.on_quote("USD_JPY", 150.09, 150.11, "t2")  # ask crosses
    assert events[0]["event"] == "FILL_LIMIT"
    assert events[0]["price"] == 150.11  # level or worse, never better


def test_stop_order_gap_fills_worse(broker):
    broker.stop_order("USD_JPY", "SHORT", 10_000, price=149.90)
    events = broker.on_quote("USD_JPY", 149.80, 149.82, "t1")  # gap through
    assert events[0]["price"] == 149.80  # the worse real bid


def test_slippage_is_adverse_on_tp_sl_and_manual_close(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
        slippage_pips=1.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")

    tp_trade = b.market_order("USD_JPY", "LONG", 1_000, tp_pips=5)
    assert b.positions[tp_trade].entry_price == 150.03
    tp_events = b.on_quote("USD_JPY", 150.08, 150.10, "2026-01-01T00:01:00+00:00#O")
    assert tp_events[0]["event"] == "EXIT_TP"
    assert tp_events[0]["price"] == 150.08
    assert tp_events[0]["price_protection"] is True
    assert tp_events[0]["applied_slippage_pips"] == 0.0

    sl_trade = b.market_order("USD_JPY", "SHORT", 1_000, sl_pips=5)
    assert b.positions[sl_trade].entry_price == 150.07
    sl_events = b.on_quote("USD_JPY", 150.12, 150.14, "2026-01-01T00:02:00+00:00#O")
    assert sl_events[0]["event"] == "EXIT_SL"
    assert sl_events[0]["price"] == 150.15

    manual_trade = b.market_order("USD_JPY", "LONG", 1_000)
    b.on_quote("USD_JPY", 150.20, 150.22, "2026-01-01T00:03:00+00:00#O")
    b.close_trade(manual_trade)
    close = [
        json.loads(line)
        for line in b.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "CLOSE"
    ][-1]["payload"]
    assert close["price"] == 150.19


def test_margin_closeout_applies_slippage_and_financing(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=200_000.0,
        slippage_pips=1.0,
        financing_pips_per_day=1.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")
    b.market_order("USD_JPY", "LONG", 30_000)
    events = b.on_quote("USD_JPY", 145.50, 145.52, "2026-01-02T00:00:00+00:00#O")
    closeout = next(event for event in events if event["event"] == "MARGIN_CLOSEOUT")
    assert closeout["price"] == 145.49
    assert closeout["financing_jpy"] == pytest.approx(300.0)
    expected = (145.49 - 150.03) * 30_000 - 300.0
    assert closeout["pl_jpy"] == pytest.approx(expected, abs=0.01)


def test_fill_ledger_binds_conversion_quote_and_never_uses_future_quote(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O")
    b.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00#O")
    trade_id = b.market_order("EUR_USD", "LONG", 10_000)

    # This conversion quote arrives after the EUR/USD fill quote. Closing at
    # the still-current EUR/USD quote must use the latest conversion at or
    # before that quote, never this future same-epoch phase.
    b.on_quote("USD_JPY", 151.00, 151.02, "2026-01-01T00:00:00+00:00#C")
    b.close_trade(trade_id)

    records = [json.loads(line) for line in b.ledger_path.read_text().splitlines()]
    fills = [rec for rec in records if rec["event"] in {"FILL_MARKET", "CLOSE"}]
    assert len(fills) == 2
    for rec in fills:
        conversion = rec["payload"]["conversion"]
        assert conversion["rate_jpy_per_quote_unit"] == pytest.approx(150.01)
        assert conversion["source_quotes"] == [
            {
                "pair": "USD_JPY",
                "bid": 150.00,
                "ask": 150.02,
                "ts": "2026-01-01T00:00:00+00:00#O",
                "phase": "O",
            }
        ]


def test_accrued_financing_reduces_open_equity_and_margin_headroom(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
        financing_pips_per_day=10.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.market_order("USD_JPY", "LONG", 1_000)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-02T00:00:00+00:00")

    account = b.account()
    assert account["accrued_financing_jpy"] == pytest.approx(100.0)
    assert account["equity_jpy"] == pytest.approx(2_000_000 - 20 - 100)


def test_sparse_other_pair_quote_advances_weekend_financing_clock(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
        financing_pips_per_day=10.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-02T21:00:00+00:00")
    b.market_order("USD_JPY", "LONG", 1_000)

    # USD/JPY remains sparse across the weekend; an observed quote for another
    # pair still advances the causal account-valuation clock to Sunday open.
    b.on_quote("EUR_JPY", 175.00, 175.02, "2026-01-04T21:00:00+00:00")

    account = b.account()
    assert account["accrued_financing_jpy"] == pytest.approx(200.0)
    assert account["equity_jpy"] == pytest.approx(2_000_000 - 20 - 200)


def test_non_finite_actions_and_snapshot_fail_closed(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    trade_id = b.market_order("USD_JPY", "LONG", 1_000)
    balance = b.balance_jpy
    units = b.positions[trade_id].units

    with pytest.raises(VirtualBrokerError, match="finite"):
        b.close_trade(trade_id, units=float("nan"))
    with pytest.raises(VirtualBrokerError, match="finite"):
        b.set_exit(trade_id, tp_price=float("inf"))
    with pytest.raises(VirtualBrokerError, match="finite"):
        b.limit_order("USD_JPY", "LONG", 1_000, price=float("nan"))
    assert b.balance_jpy == balance
    assert b.positions[trade_id].units == units
    assert "NaN" not in b.ledger_path.read_text()

    poisoned = b.snapshot()
    poisoned["balance_jpy"] = float("nan")
    clean = VirtualBroker(ledger_path=tmp_path / "clean.jsonl")
    with pytest.raises(VirtualBrokerError, match="non-finite"):
        clean.restore(poisoned)


def test_stress_slippage_cannot_execute_through_limit_price(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
        slippage_pips=2.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.limit_order("USD_JPY", "LONG", 1_000, price=149.95)
    event = b.on_quote("USD_JPY", 149.92, 149.94, "2026-01-01T00:01:00+00:00")[0]
    assert event["price"] == 149.95
    assert event["price"] <= 149.95
    assert event["price_protection"] is True


def test_quote_batch_uses_current_same_phase_conversion_independent_of_pair_order(
    tmp_path,
):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote_batch(
        [
            ("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00#O"),
            ("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O"),
        ]
    )
    trade_id = b.market_order("EUR_USD", "LONG", 1_000)
    b.on_quote_batch(
        [
            ("EUR_USD", 1.1010, 1.1012, "2026-01-01T00:01:00+00:00#C"),
            ("USD_JPY", 160.00, 160.02, "2026-01-01T00:01:00+00:00#C"),
        ]
    )
    b.close_trade(trade_id)

    close = next(
        json.loads(line)["payload"]
        for line in reversed(b.ledger_path.read_text().splitlines())
        if json.loads(line)["event"] == "CLOSE"
    )
    assert close["conversion"]["rate_jpy_per_quote_unit"] == pytest.approx(160.01)
    assert close["conversion"]["source_quotes"][0]["phase"] == "C"


def test_quote_batch_defers_portfolio_margin_check_until_every_pair_is_processed(
    tmp_path, monkeypatch
):
    calls = 0
    original = VirtualBroker._enforce_margin_after_action

    def counted(self, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, **kwargs)

    monkeypatch.setattr(VirtualBroker, "_enforce_margin_after_action", counted)
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)

    b.on_quote_batch(
        [
            ("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O"),
            ("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00#O"),
        ]
    )

    assert calls == 1


def test_sparse_batch_refuses_margin_closeout_at_stale_position_quote(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=200_000.0,
    )
    friday = "2026-01-02T21:00:00+00:00"
    b.on_quote_batch(
        [
            ("USD_JPY", 150.00, 150.02, friday),
            ("EUR_JPY", 175.00, 175.02, friday),
        ]
    )
    usd_trade = b.market_order("USD_JPY", "LONG", 10_000)
    eur_trade = b.market_order("EUR_JPY", "LONG", 10_000)

    # Only EUR/JPY reopens and moves adversely.  The resulting portfolio
    # closeout must not liquidate USD/JPY at its stale Friday quote.
    with pytest.raises(
        VirtualBrokerError,
        match="margin closeout requires current batch quotes.*USD_JPY",
    ):
        b.on_quote_batch(
            [
                (
                    "EUR_JPY",
                    160.00,
                    160.02,
                    "2026-01-04T21:00:00+00:00",
                )
            ]
        )

    assert set(b.positions) == {usd_trade, eur_trade}
    events = [
        json.loads(line)["event"] for line in b.ledger_path.read_text().splitlines()
    ]
    assert "MARGIN_CLOSEOUT" not in events


def test_stale_existing_position_quote_rejects_other_pair_market_entry(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
    )
    b.on_quote_batch([("USD_JPY", 150.00, 150.02, "2026-01-02T21:00:00+00:00")])
    existing_trade = b.market_order("USD_JPY", "LONG", 1_000)

    b.on_quote_batch([("EUR_JPY", 175.00, 175.02, "2026-01-04T21:00:00+00:00")])
    with pytest.raises(VirtualBrokerError, match="insufficient margin"):
        b.market_order("EUR_JPY", "LONG", 1_000)

    assert set(b.positions) == {existing_trade}


def test_future_existing_position_quote_rejects_other_pair_market_entry(tmp_path):
    b = VirtualBroker(
        ledger_path=tmp_path / "l.jsonl",
        balance_jpy=2_000_000.0,
    )
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:01:00+00:00")
    existing_trade = b.market_order("USD_JPY", "LONG", 1_000)

    # Observation order alone cannot make a wall-clock-future position mark
    # valid for an earlier action quote.
    b.on_quote("EUR_JPY", 175.00, 175.02, "2026-01-01T00:00:00+00:00")
    with pytest.raises(VirtualBrokerError, match="insufficient margin"):
        b.market_order("EUR_JPY", "LONG", 1_000)

    assert set(b.positions) == {existing_trade}


def test_quote_batch_has_canonical_pair_tie_break_independent_of_input_order(tmp_path):
    quotes = [
        ("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O"),
        ("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00#O"),
    ]
    quote_sequences = []
    for index, batch in enumerate((quotes, list(reversed(quotes)))):
        broker = VirtualBroker(
            ledger_path=tmp_path / f"batch-{index}.jsonl",
            balance_jpy=2_000_000.0,
        )
        broker.on_quote_batch(batch)
        quote_sequences.append(broker.snapshot()["last_quote_sequences"])

    assert quote_sequences == [
        {"EUR_USD": 1, "USD_JPY": 2},
        {"EUR_USD": 1, "USD_JPY": 2},
    ]


def test_conversion_freshness_accepts_fixed_sparse_900_second_boundary(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:15:00+00:00")

    trade_id = b.market_order("EUR_USD", "LONG", 1_000)

    assert trade_id in b.positions


def test_conversion_freshness_rejects_901_seconds_and_future_quotes(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:15:01+00:00")
    with pytest.raises(VirtualBrokerError, match="stale conversion"):
        b.market_order("EUR_USD", "LONG", 1_000)

    future = VirtualBroker(
        ledger_path=tmp_path / "future.jsonl", balance_jpy=2_000_000.0
    )
    future.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:01+00:00")
    future.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00")
    with pytest.raises(VirtualBrokerError, match="future conversion"):
        future.market_order("EUR_USD", "LONG", 1_000)


def test_unpriceable_margin_fails_closed(tmp_path):
    healthy = VirtualBroker(
        ledger_path=tmp_path / "healthy.jsonl", balance_jpy=2_000_000.0
    )
    healthy.on_quote("EUR_JPY", 160.00, 160.02, "2026-01-01T00:00:00+00:00")
    healthy.market_order("EUR_JPY", "LONG", 1_000)
    del healthy.last_quotes["EUR_JPY"]
    before = set(healthy.positions)
    with pytest.raises(VirtualBrokerError, match="no quote for open position"):
        healthy.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:01+00:00")
    assert set(healthy.positions) == before


def test_existing_ledger_chain_is_fully_verified_before_append(tmp_path):
    path = tmp_path / "l.jsonl"
    b = VirtualBroker(ledger_path=path)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.market_order("USD_JPY", "LONG", 1_000)
    b._handle.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["payload"]["tampered"] = True
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    with pytest.raises(VirtualBrokerError, match="ledger sha mismatch"):
        VirtualBroker(ledger_path=path)


def test_test_helper_sha_matches_ledger_body(broker):
    broker.market_order("USD_JPY", "LONG", 1_000)
    for line in broker.ledger_path.read_text().splitlines():
        record = json.loads(line)
        supplied = record.pop("sha")
        assert supplied == _sha(record)


def test_quote_batch_receipt_is_canonical_and_broker_chained(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    first_coordinate = {
        "mode": "replay",
        "epoch": 1_735_689_600,
        "phase": "O",
        "granularity": "M1",
        "intrabar": "OHLC",
    }
    first_quotes = [
        ("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O"),
        ("EUR_USD", 1.1, 1.1002, "2025-01-01T00:00:00+00:00#O"),
    ]

    first = b.record_quote_batch_begin(
        first_quotes,
        coordinate=first_coordinate,
        feed_pairs=["USD_JPY", "EUR_USD"],
    )
    second = b.record_quote_batch_begin(
        [("USD_JPY", 150.01, 150.03, "2025-01-01T00:00:00+00:00#H")],
        coordinate={**first_coordinate, "phase": "H"},
        feed_pairs=["EUR_USD", "USD_JPY"],
    )

    assert first["contract"] == "QR_VIRTUAL_QUOTE_BATCH_V1"
    assert first["batch_index"] == 0
    assert first["previous_batch_sha256"] == "0" * 64
    assert first["feed_pairs"] == ["EUR_USD", "USD_JPY"]
    assert first["batch_pairs"] == ["EUR_USD", "USD_JPY"]
    assert [row["pair"] for row in first["quotes"]] == ["EUR_USD", "USD_JPY"]
    assert first["coverage_complete"] is True
    assert first["quotes_sha256"] == _sha(first["quotes"])
    assert first["batch_sha256"] == _sha(
        {key: value for key, value in first.items() if key != "batch_sha256"}
    )
    assert second["batch_index"] == 1
    assert second["previous_batch_sha256"] == first["batch_sha256"]
    assert second["coverage_complete"] is False
    assert second["batch_pairs"] == ["USD_JPY"]

    events = [
        json.loads(line)["event"] for line in b.ledger_path.read_text().splitlines()
    ]
    assert events == ["QUOTE_BATCH_BEGIN", "QUOTE_BATCH_BEGIN"]


def test_account_marks_bind_broker_state_and_allow_terminal_final_binding(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    start = b.account_mark("START")
    coordinate = {
        "mode": "replay",
        "epoch": 1_735_689_600,
        "phase": "O",
        "granularity": "M1",
        "intrabar": "OHLC",
    }
    quotes = [
        ("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O"),
        ("EUR_USD", 1.1, 1.1002, "2025-01-01T00:00:00+00:00#O"),
    ]
    receipt = b.record_quote_batch_begin(
        quotes,
        coordinate=coordinate,
        feed_pairs=["USD_JPY", "EUR_USD"],
    )
    b.on_quote_batch(quotes)
    b.market_order("USD_JPY", "LONG", 100)
    b.limit_order("EUR_USD", "SHORT", 100, price=1.2, tp_pips=10, sl_pips=5)
    b.feed_cursor = {
        "mode": "replay",
        "epoch": coordinate["epoch"],
        "phase": "O",
        "bar_count": 0,
        "completed": False,
        "replay_identity_sha256": "a" * 64,
    }
    phase = b.account_mark("PHASE", batch_receipt=receipt)
    terminal_cursor = {**b.feed_cursor, "completed": True}
    terminal = b.account_mark(
        "TERMINAL", batch_receipt=receipt, feed_cursor=terminal_cursor
    )

    assert start["mark_index"] == 0
    assert start["coordinate"] is None
    assert start["batch_index"] is None
    assert start["batch_sha256"] is None
    assert start["feed_cursor"] is None
    assert start["previous_mark_sha256"] == "0" * 64
    assert start["mark_sha256"] == _sha(
        {key: value for key, value in start.items() if key != "mark_sha256"}
    )

    assert phase["mark_index"] == 1
    assert phase["previous_mark_sha256"] == start["mark_sha256"]
    assert phase["coordinate"] == coordinate
    assert phase["batch_index"] == receipt["batch_index"]
    assert phase["batch_sha256"] == receipt["batch_sha256"]
    assert [row["trade_id"] for row in phase["positions"]] == ["T000001"]
    assert [row["order_id"] for row in phase["orders"]] == ["O000002"]
    assert [row["pair"] for row in phase["quotes"]] == ["EUR_USD", "USD_JPY"]
    assert phase["account"] == b.account()
    for field in ("account", "positions", "orders", "quotes"):
        assert phase[f"{field}_sha256"] == _sha(phase[field])
    assert phase["mark_sha256"] == _sha(
        {key: value for key, value in phase.items() if key != "mark_sha256"}
    )

    assert terminal["kind"] == "TERMINAL"
    assert terminal["mark_index"] == 2
    assert terminal["previous_mark_sha256"] == phase["mark_sha256"]
    assert terminal["coordinate"] is None
    assert terminal["batch_index"] == receipt["batch_index"]
    assert terminal["batch_sha256"] == receipt["batch_sha256"]
    assert terminal["feed_cursor"] == terminal_cursor
    assert terminal["account"] == b.account()


def test_account_mark_rejects_tamper_phase_reuse_and_post_terminal_work(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    b.account_mark("START")
    coordinate = _replay_coordinate()
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]
    receipt = b.record_quote_batch_begin(
        quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
    )
    b.on_quote_batch(quotes)
    b.feed_cursor = {
        "mode": "replay",
        "epoch": coordinate["epoch"],
        "phase": "O",
    }
    tampered = json.loads(json.dumps(receipt))
    tampered["quotes"][0]["ask"] = 999.0

    with pytest.raises(VirtualBrokerError, match="digest mismatch"):
        b.account_mark("PHASE", batch_receipt=tampered)
    phase = b.account_mark("PHASE", batch_receipt=receipt)
    with pytest.raises(VirtualBrokerError, match="reused or skips"):
        b.account_mark("PHASE", batch_receipt=receipt)

    b.feed_cursor["completed"] = True
    terminal = b.account_mark("TERMINAL", batch_receipt=receipt)
    assert terminal["previous_mark_sha256"] == phase["mark_sha256"]
    with pytest.raises(VirtualBrokerError, match="follow TERMINAL"):
        b.account_mark("TERMINAL", batch_receipt=receipt)
    with pytest.raises(VirtualBrokerError, match="follow a terminal"):
        b.record_quote_batch_begin(
            quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
        )


def test_terminal_requires_latest_phase_receipt_and_completed_matching_cursor(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    b.account_mark("START")
    coordinate = _replay_coordinate()
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]
    receipt = b.record_quote_batch_begin(
        quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
    )
    b.on_quote_batch(quotes)
    b.feed_cursor = {
        "mode": "replay",
        "epoch": coordinate["epoch"],
        "phase": "O",
        "completed": False,
    }
    b.account_mark("PHASE", batch_receipt=receipt)

    with pytest.raises(VirtualBrokerError, match="latest PHASE-marked"):
        b.account_mark("TERMINAL")
    with pytest.raises(VirtualBrokerError, match="completed true"):
        b.account_mark("TERMINAL", batch_receipt=receipt)
    with pytest.raises(VirtualBrokerError, match="epoch does not match"):
        b.account_mark(
            "TERMINAL",
            batch_receipt=receipt,
            feed_cursor={**b.feed_cursor, "epoch": 0, "completed": True},
        )

    terminal = b.account_mark(
        "TERMINAL",
        batch_receipt=receipt,
        feed_cursor={**b.feed_cursor, "completed": True},
    )
    assert terminal["batch_sha256"] == receipt["batch_sha256"]
    assert terminal["feed_cursor"]["completed"] is True


def test_terminal_freezes_every_public_broker_state_mutation(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.account_mark("START")
    coordinate = _replay_coordinate()
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]
    receipt = b.record_quote_batch_begin(
        quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
    )
    b.on_quote_batch(quotes)
    trade_id = b.market_order("USD_JPY", "LONG", 100)
    order_id = b.limit_order("USD_JPY", "LONG", 100, 149.0)
    b.feed_cursor = {
        "mode": "replay",
        "epoch": coordinate["epoch"],
        "phase": "O",
        "completed": False,
    }
    b.account_mark("PHASE", batch_receipt=receipt)
    pre_terminal_snapshot = b.snapshot()
    b.feed_cursor["completed"] = True
    b.account_mark("TERMINAL", batch_receipt=receipt)
    frozen_snapshot = b.snapshot()

    mutations = [
        lambda: b.on_quote("USD_JPY", 150.01, 150.03, "2025-01-01T00:01:00+00:00"),
        lambda: b.on_quote_batch(
            [("USD_JPY", 150.01, 150.03, "2025-01-01T00:01:00+00:00")]
        ),
        lambda: b.market_order("USD_JPY", "LONG", 1),
        lambda: b.limit_order("USD_JPY", "LONG", 1, 149.0),
        lambda: b.stop_order("USD_JPY", "LONG", 1, 151.0),
        lambda: b.cancel_order(order_id),
        lambda: b.close_trade(trade_id),
        lambda: b.set_exit(trade_id, sl_price=149.5),
        lambda: b.restore(pre_terminal_snapshot),
        lambda: b.record_quote_batch_begin(
            quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
        ),
    ]
    for mutate in mutations:
        with pytest.raises(VirtualBrokerError, match="terminal|TERMINAL"):
            mutate()
        assert b.snapshot() == frozen_snapshot

    b._log("SESSION_STOP", {"account": b.account()})
    assert json.loads(b.ledger_path.read_text().splitlines()[-1])["event"] == (
        "SESSION_STOP"
    )


def test_account_mark_requires_each_started_batch_to_receive_one_phase(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    b.account_mark("START")
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]
    b.record_quote_batch_begin(
        quotes,
        coordinate=_replay_coordinate(),
        feed_pairs=["USD_JPY"],
    )

    with pytest.raises(VirtualBrokerError, match="no post-action PHASE"):
        b.record_quote_batch_begin(
            [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#H")],
            coordinate=_replay_coordinate("H"),
            feed_pairs=["USD_JPY"],
        )


def test_phase_mark_requires_exact_committed_quote_batch_application(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    b.account_mark("START")
    coordinate = _replay_coordinate()
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]
    receipt = b.record_quote_batch_begin(
        quotes, coordinate=coordinate, feed_pairs=["USD_JPY"]
    )
    b.feed_cursor = {
        "mode": "replay",
        "epoch": coordinate["epoch"],
        "phase": "O",
    }

    with pytest.raises(VirtualBrokerError, match="has not been applied"):
        b.account_mark("PHASE", batch_receipt=receipt)

    b.on_quote_batch(quotes)
    phase = b.account_mark("PHASE", batch_receipt=receipt)
    assert phase["quotes"][0]["sequence"] == 1
    assert phase["quotes"][0]["watermark"] == 1


def test_continuous_marks_reject_uncommitted_or_mismatched_quote_delivery(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl")
    b.account_mark("START")
    quotes = [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")]

    with pytest.raises(VirtualBrokerError, match="no preceding broker commitment"):
        b.on_quote_batch(quotes)
    with pytest.raises(VirtualBrokerError, match="atomic committed"):
        b.on_quote(*quotes[0])

    b.record_quote_batch_begin(
        quotes, coordinate=_replay_coordinate(), feed_pairs=["USD_JPY"]
    )
    with pytest.raises(VirtualBrokerError, match="does not match broker commitment"):
        b.on_quote_batch([("USD_JPY", 149.0, 149.02, "2025-01-01T00:00:00+00:00#O")])
    b.on_quote_batch(quotes)
    with pytest.raises(VirtualBrokerError, match="no unconsumed broker commitment"):
        b.on_quote_batch(quotes)


def test_continuous_evidence_index_cannot_restart_on_reopened_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    b = VirtualBroker(ledger_path=path)
    b.account_mark("START")
    b._handle.close()

    reopened = VirtualBroker(ledger_path=path)
    with pytest.raises(VirtualBrokerError, match="fresh broker session"):
        reopened.account_mark("START")
    with pytest.raises(VirtualBrokerError, match="fresh broker session"):
        reopened.record_quote_batch_begin(
            [("USD_JPY", 150.0, 150.02, "2025-01-01T00:00:00+00:00#O")],
            coordinate=_replay_coordinate(),
            feed_pairs=["USD_JPY"],
        )
