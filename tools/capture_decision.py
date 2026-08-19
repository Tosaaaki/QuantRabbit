#!/usr/bin/env python3
"""Record one operator decision as a `QR_DIRECT_MANUAL_DECISION_CAPSULE_V1`.

    tools/capture_decision.py "USDJPY skip 弱い"
    tools/capture_decision.py "GBPJPY long 0.8 確信中"
    tools/capture_decision.py --verify

The operator's label is the irreplaceable half of the record: it exists for a
few seconds and then it is gone. Broker context can always be re-derived from a
timestamp. So a broker read that fails does **not** fail the capture — the
capsule is written with null market context and the reason recorded in
`missing[]`. Losing the label because an HTTP call timed out would reproduce the
exact gap this recorder exists to close.

Read-only by construction: this tool imports `OandaReadOnlyClient` and never the
execution client.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_rabbit.decision_capsule import (  # noqa: E402
    CAPSULE_TIMEFRAMES,
    MIN_COMPLETE_BARS,
    OANDA_GRANULARITY,
    CapsuleError,
    append_capsule,
    build_broker_context,
    build_capsule,
    canonical_json,
    default_artifacts_root,
    feature_spec,
    iso,
    parse_intake,
    pip_size,
    timeframe_features,
    utc_now,
    validate_capsule,
    verify_chain,
)

SCHEMA_PATH = ROOT / "docs" / "schemas" / "manual_decision_capsule_v1.schema.json"


def _empty_timeframes(observed_at: datetime) -> list[dict]:
    return [timeframe_features(timeframe, [], "USD_JPY", observed_at) for timeframe in CAPSULE_TIMEFRAMES]


def _serialize_position(position) -> dict:
    return {
        "trade_id": position.trade_id,
        "pair": position.pair,
        "side": getattr(position.side, "value", str(position.side)),
        "units": position.units,
        "entry_price": position.entry_price,
        "unrealized_pl_jpy": position.unrealized_pl_jpy,
        "take_profit": position.take_profit,
        "stop_loss": position.stop_loss,
        "owner": getattr(position.owner, "value", str(position.owner)),
    }


def _serialize_order(order) -> dict:
    return {
        "order_id": order.order_id,
        "pair": order.pair,
        "order_type": order.order_type,
        "trade_id": order.trade_id,
        "price": order.price,
        "state": order.state,
        "units": order.units,
        "owner": getattr(order.owner, "value", str(order.owner)),
    }


def collect_market_context(pair: str, observed_at: datetime, env_file: Path | None):
    """Return (timeframes, broker_context, missing). Never raises on broker failure."""

    missing: list[dict[str, str]] = []
    try:
        from quant_rabbit.broker.oanda import OandaReadOnlyClient

        client = OandaReadOnlyClient(env_file=env_file)
    except Exception as exc:  # noqa: BLE001 - the label must survive any broker fault
        missing.append({"field": "broker_context", "reason": f"BROKER_UNAVAILABLE: {type(exc).__name__}"})
        missing.append({"field": "market_context.timeframes", "reason": f"BROKER_UNAVAILABLE: {type(exc).__name__}"})
        return _empty_timeframes(observed_at), build_broker_context(
            quote_time_utc=None, bid=None, ask=None, spread=None, nav=None,
            margin_available=None, margin_used=None, positions=None, orders=None,
            transaction_watermark=None,
        ), missing

    timeframes = []
    for timeframe in CAPSULE_TIMEFRAMES:
        try:
            payload = client.get_json(
                f"/v3/instruments/{pair}/candles",
                {
                    "granularity": OANDA_GRANULARITY[timeframe],
                    "count": str(MIN_COMPLETE_BARS + 3),
                    "price": "M",
                },
            )
            candles = payload.get("candles") or []
        except Exception as exc:  # noqa: BLE001
            candles = []
            missing.append(
                {"field": f"market_context.timeframes[{timeframe}]", "reason": f"CANDLE_FETCH_FAILED: {type(exc).__name__}"}
            )
        timeframes.append(timeframe_features(timeframe, candles, pair, observed_at))

    quote_time = bid = ask = spread = None
    nav = margin_available = margin_used = None
    positions = orders = None
    watermark = None
    try:
        snapshot = client.snapshot([pair])
        quote = snapshot.quotes.get(pair)
        if quote is not None:
            bid, ask = quote.bid, quote.ask
            quote_time = iso(quote.timestamp_utc)
            spread = round((ask - bid) / pip_size(pair), 6)
        positions = [_serialize_position(item) for item in snapshot.positions]
        orders = [_serialize_order(item) for item in snapshot.orders]
        if snapshot.account is not None:
            nav = snapshot.account.nav_jpy
            margin_available = snapshot.account.margin_available_jpy
            margin_used = snapshot.account.margin_used_jpy
    except Exception as exc:  # noqa: BLE001
        missing.append({"field": "broker_context.quote", "reason": f"SNAPSHOT_FAILED: {type(exc).__name__}"})

    try:
        summary = client.get_json(f"/v3/accounts/{client.account_id}/summary")
        watermark = summary.get("lastTransactionID")
    except Exception as exc:  # noqa: BLE001
        missing.append({"field": "broker_context.transaction_watermark", "reason": f"SUMMARY_FAILED: {type(exc).__name__}"})

    return timeframes, build_broker_context(
        quote_time_utc=quote_time, bid=bid, ask=ask, spread=spread, nav=nav,
        margin_available=margin_available, margin_used=margin_used,
        positions=positions, orders=orders, transaction_watermark=watermark,
    ), missing


def artifacts_dir(root: Path, moment: datetime) -> Path:
    return root / moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("intake", nargs="*", help="'<pair> <action> [confidence] [note...]'")
    parser.add_argument("--env-file", type=Path, default=None, help="path to .env.local holding QR_OANDA_*")
    parser.add_argument("--artifacts-root", type=Path, default=None, help="override capsule output root")
    parser.add_argument("--dry-run", action="store_true", help="build and validate, print, write nothing")
    parser.add_argument("--verify", action="store_true", help="verify today's capsule hash chain and exit")
    parser.add_argument("--json", action="store_true", help="print the full capsule instead of a one-line receipt")
    args = parser.parse_args(argv)

    now = utc_now()
    root = args.artifacts_root or default_artifacts_root()
    artifacts = artifacts_dir(root, now)

    if args.verify:
        print(json.dumps(verify_chain(artifacts), ensure_ascii=False, indent=2))
        return 0

    raw = " ".join(args.intake).strip()
    if not raw:
        parser.error("nothing to record; pass an intake line such as 'USDJPY skip 弱い'")

    try:
        intake = parse_intake(raw)
    except CapsuleError as exc:
        print(f"intake rejected: {exc}", file=sys.stderr)
        return 2

    timeframes, broker_context, missing = collect_market_context(intake.pair, now, args.env_file)
    # The cutoff is the moment of capture: every feature above is built from
    # bars that had already closed by then, so the capsule carries no future.
    capsule = build_capsule(
        intake,
        captured_at=now,
        decision_cutoff=now,
        timeframes=timeframes,
        broker_context=broker_context,
        extra_missing=missing,
    )

    try:
        validate_capsule(capsule, SCHEMA_PATH)
    except CapsuleError as exc:
        print(f"capsule rejected: {exc}", file=sys.stderr)
        return 3

    if args.dry_run:
        print(json.dumps(capsule, ensure_ascii=False, indent=2))
        return 0

    artifacts.mkdir(parents=True, exist_ok=True)
    spec_path = artifacts / "feature_spec.json"
    if not spec_path.exists():
        spec_path.write_text(canonical_json(feature_spec()) + "\n", encoding="utf-8")

    row = append_capsule(capsule, artifacts)
    if args.json:
        print(json.dumps(capsule, ensure_ascii=False, indent=2))
    else:
        evidence = capsule["operator_evidence"]
        spread = broker_context["spread"]
        print(
            f"recorded {capsule['capsule_id'][:16]}… "
            f"{capsule['pair']} {evidence['primary_action']}"
            f"{'/' + evidence['side'] if evidence['side'] else ''} "
            f"conf={evidence['confidence']} "
            f"spread={spread if spread is not None else 'null'}pips "
            f"missing={len(capsule['missing'])} -> {artifacts}"
        )
    return 0 if row else 1


if __name__ == "__main__":
    raise SystemExit(main())
