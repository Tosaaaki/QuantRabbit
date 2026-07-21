import json

import pytest

from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


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
    broker.on_quote("EUR_USD", 1.1000, 1.1002, "t0")
    tid = broker.market_order("EUR_USD", "LONG", 10_000, tp_pips=10)
    events = broker.on_quote("EUR_USD", 1.1012, 1.1014, "t1")
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


def test_existing_ledger_hash_corruption_fails_closed(broker):
    broker.market_order("USD_JPY", "LONG", 1_000)
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    records[0]["payload"]["tampered"] = True
    broker._handle.close()
    broker.ledger_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )

    with pytest.raises(VirtualBrokerError, match="ledger sha mismatch"):
        VirtualBroker(broker.ledger_path)


def test_invalid_quote_refused(broker):
    with pytest.raises(VirtualBrokerError, match="invalid quote"):
        broker.on_quote("USD_JPY", 150.05, 150.01, "t1")  # ask < bid


def test_snapshot_restore_roundtrip(broker, tmp_path):
    broker.market_order("USD_JPY", "LONG", 10_000, tp_pips=5)
    broker.limit_order("USD_JPY", "SHORT", 5_000, price=150.50)
    snap = broker.snapshot()
    b2 = VirtualBroker(ledger_path=tmp_path / "l2.jsonl")
    b2.restore(snap)
    b2.on_quote("USD_JPY", 150.00, 150.02, "t0")
    assert b2.account()["open_positions"] == 1
    assert b2.account()["resting_orders"] == 1
    assert b2.balance_jpy == broker.balance_jpy


def test_strategy_tag_survives_order_fill_exit_ledger_and_snapshot(broker):
    oid = broker.limit_order(
        "USD_JPY", "LONG", 1_000, price=149.95, tp_pips=5,
        strategy_tag="W_FADE",
    )
    order = broker.orders[oid]
    assert order.strategy_tag == "W_FADE"

    fill = broker.on_quote("USD_JPY", 149.92, 149.94, "t1")[0]
    trade_id = fill["trade_id"]
    assert fill["strategy_tag"] == "W_FADE"
    assert broker.positions[trade_id].strategy_tag == "W_FADE"
    assert broker.snapshot()["positions"][0]["strategy_tag"] == "W_FADE"

    exit_event = broker.on_quote("USD_JPY", 149.99, 150.01, "t2")[0]
    assert exit_event["event"] == "EXIT_TP"
    assert exit_event["strategy_tag"] == "W_FADE"
    records = [json.loads(line) for line in broker.ledger_path.read_text().splitlines()]
    tagged = [record for record in records if record["payload"].get("strategy_tag")]
    assert {record["payload"]["strategy_tag"] for record in tagged} == {"W_FADE"}


def test_snapshot_ledger_checkpoint_mismatch_fails_closed(broker, tmp_path):
    broker.market_order("USD_JPY", "LONG", 1_000)
    snap = broker.snapshot()
    other = VirtualBroker(ledger_path=tmp_path / "other.jsonl")

    with pytest.raises(VirtualBrokerError, match="terminal sha"):
        other.restore(snap, require_ledger_match=True)


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
