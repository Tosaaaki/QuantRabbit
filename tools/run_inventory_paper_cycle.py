#!/usr/bin/env python3
"""Run one GET-quote fast-bot/LLM/inventory paper cycle without gateway claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from quant_rabbit.broker.execution import _oanda_order_request
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot import REGIME_CONTRACT, _seal, build_fast_bot_shadow
from quant_rabbit.inventory_controller import InventoryController, LotIdentity
from quant_rabbit.models import OrderIntent, OrderType, Quote, Side


def run_paper_cycle(*, quotes: Mapping[str, Quote], now_utc: datetime) -> dict[str, object]:
    rows = [
        {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "RANGE_ROTATION",
            "state": "GO",
            "execution_enabled": True,
            "score": 5.0,
            "m1_closed_candle_utc": now_utc.isoformat(),
            "m5_atr_pips": 5.0,
        },
        {
            "pair": "USD_JPY",
            "side": "SHORT",
            "method": "TREND_CONTINUATION",
            "state": "GO",
            "execution_enabled": True,
            "score": 5.0,
            "m1_closed_candle_utc": now_utc.isoformat(),
            "m5_atr_pips": 5.0,
        },
    ]
    regime = _seal(
        {
            "contract": REGIME_CONTRACT,
            "schema_version": 1,
            "generated_at_utc": now_utc.isoformat(),
            "rows": rows,
        }
    )
    snapshot = {
        "fetched_at_utc": now_utc.isoformat(),
        "quotes": {
            pair: {
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in quotes.items()
        },
    }
    shadow = build_fast_bot_shadow(regime, broker_snapshot=snapshot, now_utc=now_utc)
    signals = list(shadow["signals"])
    if len(signals) != 2:
        raise RuntimeError("paper loop requires one fresh signal for each screened pair")

    feature_snapshot = {
        "regime_contract_sha256": regime["contract_sha256"],
        "quote_timestamps": {
            pair: quote.timestamp_utc.isoformat() for pair, quote in quotes.items()
        },
    }
    feature_sha = hashlib.sha256(
        json.dumps(feature_snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    allow_event = {"event_id": "regime-ready", "dedupe_key": "PORTFOLIO|REGIME_READY"}
    allow_receipt = {
        "receipt_id": "allow-current-regime",
        "event_id": allow_event["event_id"],
        "dedupe_key": allow_event["dedupe_key"],
        "feature_snapshot_sha256": feature_sha,
        "decision": "ALLOW",
        "regime": "MIXED",
        "allowed_strategy_ids": [
            "range_rotation",
            "trend_continuation",
        ],
        "risk_budget_cap_jpy": 500.0,
        "max_positions_cap": 2,
        "generated_at_utc": now_utc.isoformat(),
        "expires_at_utc": (now_utc + timedelta(minutes=5)).isoformat(),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        stale_controller = InventoryController.open(
            Path(temp_dir) / "stale_inventory.json",
            campaign_id=str(signals[0]["campaign_id"]),
            now_utc=now_utc,
        )
        stale_result = stale_controller.apply_supervision_receipt(
            event=allow_event,
            receipt={
                **allow_receipt,
                "receipt_id": "stale-allow-current-regime",
                "generated_at_utc": (now_utc - timedelta(minutes=10)).isoformat(),
                "expires_at_utc": (now_utc - timedelta(minutes=5)).isoformat(),
            },
            now_utc=now_utc,
        )
        controller = InventoryController.open(
            Path(temp_dir) / "inventory.json",
            campaign_id=str(signals[0]["campaign_id"]),
            now_utc=now_utc,
        )
        allow_result = controller.apply_supervision_receipt(
            event=allow_event,
            receipt=allow_receipt,
            now_utc=now_utc,
        )
        duplicate_result = controller.apply_supervision_receipt(
            event=allow_event,
            receipt=allow_receipt,
            now_utc=now_utc,
        )
        staged_orders: list[dict[str, object]] = []
        for signal in signals:
            identity = LotIdentity.from_metadata(signal)
            intent = OrderIntent(
                pair=str(signal["pair"]),
                side=Side.parse(str(signal["side"])),
                order_type=OrderType.LIMIT,
                units=10,
                entry=float(signal["entry"]),
                tp=float(signal["take_profit"]),
                sl=float(signal["stop_loss"]),
                thesis="paper fast-bot proposal under bounded LLM supervision",
                metadata={
                    **identity.to_metadata(),
                    "lane_id": f"fast_bot:{signal['pair']}:{signal['side']}:{signal['method']}",
                },
            )
            order = _oanda_order_request(intent)
            staged_orders.append(order)
            controller.register_fill(
                identity=identity,
                pair=intent.pair,
                side=intent.side,
                units=intent.units,
                entry_price=float(signal["entry"]),
                now_utc=now_utc + timedelta(seconds=1),
            )
            factor = 100 if intent.pair.endswith("_JPY") else 10_000
            direction = 1 if intent.side is Side.LONG else -1
            favorable = float(signal["entry"]) + direction * (5 / factor)
            giveback = float(signal["entry"]) + direction * (2 / factor)
            controller.mark_lot(
                identity.lot_id,
                executable_price=favorable,
                now_utc=now_utc + timedelta(seconds=2),
            )
            controller.mark_lot(
                identity.lot_id,
                executable_price=giveback,
                now_utc=now_utc + timedelta(seconds=3),
            )
            controller.update_unwind_economics(
                identity.lot_id,
                estimated_margin_relief_jpy=1_000.0,
                estimated_close_loss_and_cost_jpy=10.0,
                currency_factor="USD",
                now_utc=now_utc + timedelta(seconds=3),
            )

        unwind_event = {
            "event_id": "inventory-band-breach",
            "dedupe_key": "PORTFOLIO|INVENTORY_BAND_BREACH",
        }
        unwind_receipt = {
            "receipt_id": "unwind-current-inventory",
            "event_id": unwind_event["event_id"],
            "dedupe_key": unwind_event["dedupe_key"],
            "feature_snapshot_sha256": feature_sha,
            "decision": "UNWIND",
            "regime": "MIXED",
            "allowed_strategy_ids": [],
            "risk_budget_cap_jpy": 0.0,
            "max_positions_cap": 0,
            "generated_at_utc": (now_utc + timedelta(seconds=4)).isoformat(),
            "expires_at_utc": (now_utc + timedelta(minutes=5)).isoformat(),
        }
        unwind_result = controller.apply_supervision_receipt(
            event=unwind_event,
            receipt=unwind_receipt,
            now_utc=now_utc + timedelta(seconds=4),
        )
        first_actions = controller.unwind_actions(
            now_utc=now_utc + timedelta(seconds=5),
            terminal_deadline_utc=now_utc + timedelta(minutes=1),
        )
        for action in first_actions:
            if action.action == "REDUCE_BOT_LOT":
                controller.record_unwind_fill(
                    str(action.lot_id),
                    units=int(action.units or 0),
                    realized_after_cost_jpy=0.0,
                    now_utc=now_utc + timedelta(seconds=6),
                )
        terminal_actions = controller.unwind_actions(
            now_utc=now_utc + timedelta(minutes=1),
            terminal_deadline_utc=now_utc + timedelta(minutes=1),
        )
        for action in terminal_actions:
            if action.action == "REDUCE_BOT_LOT":
                controller.record_unwind_fill(
                    str(action.lot_id),
                    units=int(action.units or 0),
                    realized_after_cost_jpy=0.0,
                    now_utc=now_utc + timedelta(minutes=1),
                )
        controller.stop(
            now_utc=now_utc + timedelta(minutes=1), cooldown=timedelta(minutes=30)
        )
        return {
            "contract": "QR_INVENTORY_PAPER_CYCLE_READBACK_V2",
            "status": "PAPER_LOOP_FLAT",
            "broker_http_methods_used": ["GET"],
            "broker_write_performed": False,
            "orders_sent": 0,
            "pairs": [signal["pair"] for signal in signals],
            "same_bid_ask_signal_and_stage": True,
            "fast_bot_signal_count": len(signals),
            "allow_receipt_result": allow_result,
            "duplicate_effective_applications": 0
            if duplicate_result == "DUPLICATE_IGNORED"
            else 1,
            "stale_decision_applications": (
                0 if stale_result == "FREEZE_NEW_INVALID_RECEIPT" else 1
            ),
            "stale_receipt_result": stale_result,
            "supervision_regime": controller.supervision_regime,
            "supervision_risk_budget_cap_jpy": (
                controller.supervision_risk_budget_cap_jpy
            ),
            "supervision_max_positions_cap": controller.supervision_max_positions_cap,
            "unwind_receipt_result": unwind_result,
            "canonical_order_request_count": len(staged_orders),
            "canonical_order_client_ids": [
                order["clientExtensions"]["id"] for order in staged_orders
            ],
            "live_order_gateway_invocation_count": 0,
            "external_mutation_gateway": "LiveOrderGateway",
            "partial_scale_out_lot_count": len(
                [action for action in first_actions if action.action == "REDUCE_BOT_LOT"]
            ),
            "terminal_liquidation_lot_count": len(
                [action for action in terminal_actions if action.action == "REDUCE_BOT_LOT"]
            ),
            "final_inventory_state": controller.state.value,
            "remaining_bot_owned_units": sum(
                lot.remaining_units for lot in controller.lots.values()
            ),
            "cooldown_until_utc": controller.cooldown_until_utc,
            "transition_events": [event["event_type"] for event in controller.events],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    client = OandaReadOnlyClient(env_file=args.env_file)
    quotes = client.quotes(("EUR_USD", "USD_JPY"))
    now = datetime.now(timezone.utc)
    result = run_paper_cycle(quotes=quotes, now_utc=now)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
