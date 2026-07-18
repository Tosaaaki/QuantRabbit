import json

import pytest

from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError, _sha


@pytest.fixture()
def broker(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "ledger.jsonl", balance_jpy=200_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "t0")
    return b


def test_market_fill_at_real_ask_and_bid(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=5, sl_pips=None)
    pos = broker.positions[tid]
    assert pos.entry_price == 150.02  # ask
    assert pos.tp_price == pytest.approx(150.07)
    sid = broker.market_order("USD_JPY", "SHORT", 5_000)
    assert broker.positions[sid].entry_price == 150.00  # bid


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
    broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=2, sl_pips=2)
    # a quote where bid is below SL (gap through both is impossible on one
    # quote; SL-first rule applies when SL is touched)
    events = broker.on_quote("USD_JPY", 149.99, 150.01, "t1")
    assert events[0]["event"] == "EXIT_SL"


def test_sl_gap_fills_at_worse_price(broker):
    tid = broker.market_order("USD_JPY", "LONG", 10_000, sl_pips=2)
    events = broker.on_quote("USD_JPY", 149.90, 149.92, "t1")  # gap through SL
    assert events[0]["event"] == "EXIT_SL"
    assert events[0]["price"] == 149.90  # the worse real bid, not the SL level


def test_limit_fill_at_level_or_better_never_synthesized(broker):
    oid = broker.limit_order("USD_JPY", "LONG", 10_000, price=149.95, tp_pips=10)
    assert not broker.on_quote("USD_JPY", 149.96, 149.98, "t1")
    events = broker.on_quote("USD_JPY", 149.92, 149.94, "t2")  # ask below limit
    assert events[0]["event"] == "FILL_LIMIT"
    assert events[0]["price"] == 149.94  # better real ask, not the level
    assert oid not in broker.orders


def test_hedge_netting_margin(broker):
    broker.market_order("USD_JPY", "LONG", 10_000)
    acct_one = broker.account()
    broker.market_order("USD_JPY", "SHORT", 10_000)
    acct_hedged = broker.account()
    # opposite position adds no margin (max of sides unchanged)
    assert acct_hedged["margin_used_jpy"] == pytest.approx(
        acct_one["margin_used_jpy"], rel=1e-6)


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
    assert broker.balance_jpy - 200_000 == pytest.approx(0.001 * 10_000 * 150.01, rel=1e-3)


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
    tp_events = b.on_quote(
        "USD_JPY", 150.08, 150.10, "2026-01-01T00:01:00+00:00#O"
    )
    assert tp_events[0]["event"] == "EXIT_TP"
    assert tp_events[0]["price"] == 150.08
    assert tp_events[0]["price_protection"] is True
    assert tp_events[0]["applied_slippage_pips"] == 0.0

    sl_trade = b.market_order("USD_JPY", "SHORT", 1_000, sl_pips=5)
    assert b.positions[sl_trade].entry_price == 150.07
    sl_events = b.on_quote(
        "USD_JPY", 150.12, 150.14, "2026-01-01T00:02:00+00:00#O"
    )
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
    events = b.on_quote(
        "USD_JPY", 145.50, 145.52, "2026-01-02T00:00:00+00:00#O"
    )
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
        assert conversion["source_quotes"] == [{
            "pair": "USD_JPY",
            "bid": 150.00,
            "ask": 150.02,
            "ts": "2026-01-01T00:00:00+00:00#O",
            "phase": "O",
        }]


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
    event = b.on_quote(
        "USD_JPY", 149.92, 149.94, "2026-01-01T00:01:00+00:00"
    )[0]
    assert event["price"] == 149.95
    assert event["price"] <= 149.95
    assert event["price_protection"] is True


def test_quote_batch_uses_current_same_phase_conversion_independent_of_pair_order(
    tmp_path,
):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote_batch([
        ("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:00:00+00:00#O"),
        ("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00#O"),
    ])
    trade_id = b.market_order("EUR_USD", "LONG", 1_000)
    b.on_quote_batch([
        ("EUR_USD", 1.1010, 1.1012, "2026-01-01T00:01:00+00:00#C"),
        ("USD_JPY", 160.00, 160.02, "2026-01-01T00:01:00+00:00#C"),
    ])
    b.close_trade(trade_id)

    close = next(
        json.loads(line)["payload"]
        for line in reversed(b.ledger_path.read_text().splitlines())
        if json.loads(line)["event"] == "CLOSE"
    )
    assert close["conversion"]["rate_jpy_per_quote_unit"] == pytest.approx(160.01)
    assert close["conversion"]["source_quotes"][0]["phase"] == "C"


def test_stale_conversion_and_unpriceable_margin_fail_closed(tmp_path):
    b = VirtualBroker(ledger_path=tmp_path / "l.jsonl", balance_jpy=2_000_000.0)
    b.on_quote("USD_JPY", 150.00, 150.02, "2026-01-01T00:00:00+00:00")
    b.on_quote("EUR_USD", 1.1000, 1.1002, "2026-01-01T00:02:00+00:00")
    with pytest.raises(VirtualBrokerError, match="stale conversion"):
        b.market_order("EUR_USD", "LONG", 1_000)

    healthy = VirtualBroker(
        ledger_path=tmp_path / "healthy.jsonl", balance_jpy=2_000_000.0
    )
    healthy.on_quote("EUR_JPY", 160.00, 160.02, "2026-01-01T00:00:00+00:00")
    healthy.market_order("EUR_JPY", "LONG", 1_000)
    del healthy.last_quotes["EUR_JPY"]
    before = set(healthy.positions)
    with pytest.raises(VirtualBrokerError, match="no quote for open position"):
        healthy.on_quote(
            "USD_JPY", 150.00, 150.02, "2026-01-01T00:00:01+00:00"
        )
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
