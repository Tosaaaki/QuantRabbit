#!/usr/bin/env python3
"""Run synchronized Paper-only champion, inventory, and challenger lanes.

One read-only OANDA quote batch is fanned out to every isolated VirtualBroker.
The process has no production broker/order client and no live promotion path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status  # noqa: E402
from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.dojo_bot_catalog import validate_bot_config  # noqa: E402
from quant_rabbit.paper_champion_challenger import (  # noqa: E402
    candidate_hash,
    canonical_sha256,
)
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError  # noqa: E402

UTC = timezone.utc
AUTHORITY = {
    "live_permission": False,
    "broker_mutation_allowed": False,
    "order_authority": "NONE",
    "auto_live_promotion": False,
}


class ExperimentConfigError(ValueError):
    """Raised before any lane starts when the registry is unsafe."""


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_lab_bot():
    path = REPO_ROOT / "bots/lab_bot.py"
    spec = importlib.util.spec_from_file_location("paper_shared_lab_bot", path)
    if spec is None or spec.loader is None:
        raise ExperimentConfigError("cannot load reviewed lab bot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Bot


def _validate_lane_config(config: dict[str, Any]) -> None:
    catalog_config = dict(config)
    catalog_config.pop("strategy_owner_id", None)
    catalog_config.pop("entry_direction_policy", None)
    validate_bot_config(catalog_config)


def load_experiment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExperimentConfigError("experiment must be a JSON object")
    if value.get("contract") != "QR_SHARED_CAUSAL_PAPER_EXPERIMENT_V1":
        raise ExperimentConfigError("unsupported experiment contract")
    if value.get("authority") != AUTHORITY:
        raise ExperimentConfigError("Paper authority invariant failed")
    if value.get("dojo_dependency") != "NONE":
        raise ExperimentConfigError("DOJO must not be a dependency")
    if value.get("future_data_allowed") is not False:
        raise ExperimentConfigError("future data must be forbidden")
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or not pairs or len(pairs) != len(set(pairs)):
        raise ExperimentConfigError("pairs must be unique and non-empty")
    lanes = value.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 6:
        raise ExperimentConfigError("exactly three roles x two cost arms are required")
    lane_ids = [lane.get("lane_id") for lane in lanes if isinstance(lane, dict)]
    if len(lane_ids) != 6 or len(set(lane_ids)) != 6:
        raise ExperimentConfigError("lane ids must be unique")
    expected = {
        ("CHAMPION", "BASE"),
        ("CHAMPION", "STRESS"),
        ("AI_INVENTORY", "BASE"),
        ("AI_INVENTORY", "STRESS"),
        ("CHALLENGER", "BASE"),
        ("CHALLENGER", "STRESS"),
    }
    actual = {(lane.get("role"), lane.get("cost_arm")) for lane in lanes}
    if actual != expected:
        raise ExperimentConfigError("role/cost arm matrix is incomplete")
    for lane in lanes:
        config = lane.get("bot_config")
        if not isinstance(config, dict):
            raise ExperimentConfigError("lane bot_config is required")
        if config.get("pairs") != pairs:
            raise ExperimentConfigError("lane pairs must match feed pairs")
        if {
            key: config.get(key)
            for key in (
                "live_permission",
                "external_broker_mutation_allowed",
                "order_authority",
            )
        } != {
            "live_permission": False,
            "external_broker_mutation_allowed": False,
            "order_authority": "NONE",
        }:
            raise ExperimentConfigError("lane authority is invalid")
        _validate_lane_config(config)
        if float(lane.get("virtual_capital_jpy", 0)) != 50_000:
            raise ExperimentConfigError("every comparison lane must use JPY 50,000")
        if not 0 < float(lane.get("max_drawdown_fraction", 0)) <= 0.05:
            raise ExperimentConfigError("lane DD kill exceeds 5%")
    candidate = value.get("candidate")
    if not isinstance(candidate, dict):
        raise ExperimentConfigError("candidate is required")
    if candidate_hash(candidate) != candidate.get("candidate_hash"):
        raise ExperimentConfigError("candidate hash mismatch")
    expected_feed_contract = canonical_sha256(
        {
            "contract": "QR_SHARED_CAUSAL_PAPER_FEED_V1",
            "pairs": pairs,
            "future_data_allowed": False,
            "single_quote_batch_fanned_out": True,
        }
    )
    if candidate.get("shared_feed_contract_sha256") != expected_feed_contract:
        raise ExperimentConfigError("candidate shared-feed contract mismatch")
    candidate_config = validate_bot_config(candidate.get("bot_config") or {})
    for lane in lanes:
        if lane["role"] != "CHALLENGER":
            continue
        lane_config = dict(lane["bot_config"])
        lane_config.pop("strategy_owner_id", None)
        if validate_bot_config(lane_config) != candidate_config:
            raise ExperimentConfigError("challenger lane differs from sealed candidate")
    return value


def _parse_stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise VirtualBrokerError("OANDA candle timestamp lacks timezone")
    return parsed.astimezone(UTC)


def seed_bots(client, bots: list[Any], pairs: list[str], count: int) -> dict[str, Any]:
    receipts = []
    now = datetime.now(UTC)
    for pair in pairs:
        payload = client.get_json(
            f"/v3/instruments/{pair}/candles",
            {"price": "BA", "granularity": "M1", "count": str(count)},
        )
        candles = payload.get("candles")
        if not isinstance(candles, list):
            raise VirtualBrokerError(f"missing OANDA seed candles: {pair}")
        bars = []
        for raw in candles:
            if not isinstance(raw, dict) or raw.get("complete") is not True:
                continue
            stamp = _parse_stamp(str(raw["time"]))
            if stamp >= now:
                raise VirtualBrokerError("future seed candle rejected")
            bid, ask = raw["bid"], raw["ask"]
            bars.append(
                {
                    "epoch": int(stamp.timestamp()),
                    "bid_o": float(bid["o"]),
                    "bid_h": float(bid["h"]),
                    "bid_l": float(bid["l"]),
                    "bid_c": float(bid["c"]),
                    "ask_o": float(ask["o"]),
                    "ask_h": float(ask["h"]),
                    "ask_l": float(ask["l"]),
                    "ask_c": float(ask["c"]),
                }
            )
        bars.sort(key=lambda row: row["epoch"])
        if len(bars) < 1441 or len({row["epoch"] for row in bars}) != len(bars):
            raise VirtualBrokerError(f"insufficient or duplicate seed: {pair}")
        if now.timestamp() - bars[-1]["epoch"] > 180:
            raise VirtualBrokerError(f"stale OANDA seed: {pair}")
        for bot in bots:
            for bar in bars:
                bot.seed_bar(pair, dict(bar))
        receipts.append(
            {
                "pair": pair,
                "bar_count": len(bars),
                "first_epoch": bars[0]["epoch"],
                "last_epoch": bars[-1]["epoch"],
                "bars_sha256": canonical_sha256(bars),
            }
        )
    body = {
        "contract": "QR_SHARED_PAPER_SEED_V1",
        "pairs": receipts,
        "single_seed_fanned_out": True,
        "authority": AUTHORITY,
    }
    body["seed_sha256"] = canonical_sha256(body)
    return body


def _lane_state(lane: dict[str, Any]) -> dict[str, Any]:
    broker: VirtualBroker = lane["broker"]
    account = broker.account()
    peak = max(float(lane["peak_equity_jpy"]), float(account["equity_jpy"]))
    lane["peak_equity_jpy"] = peak
    drawdown = (peak - float(account["equity_jpy"])) / float(
        lane["virtual_capital_jpy"]
    )
    return {
        "lane_id": lane["lane_id"],
        "role": lane["role"],
        "cost_arm": lane["cost_arm"],
        "active": lane["active"],
        "stop_reason": lane.get("stop_reason"),
        "peak_equity_jpy": round(peak, 2),
        "drawdown_fraction": round(drawdown, 8),
        "account": account,
        "positions": [vars(value) for value in broker.positions.values()],
        "orders": [vars(value) for value in broker.orders.values()],
        "ledger_tip_sha256": broker.snapshot()["ledger_tip_sha"],
    }


def _persist(root: Path, lanes: list[dict[str, Any]], coordinator: dict[str, Any]) -> None:
    states = []
    for lane in lanes:
        lane_root = root / lane["lane_id"]
        state = _lane_state(lane)
        _atomic_json(lane_root / "broker_snapshot.json", lane["broker"].snapshot())
        _atomic_json(lane_root / "state.json", state)
        states.append(state)
    _atomic_json(root / "coordinator_state.json", {**coordinator, "lanes": states})


def _kill_if_needed(lane: dict[str, Any]) -> None:
    if not lane["active"] or lane["role"] != "CHALLENGER":
        return
    account = lane["broker"].account()
    peak = max(float(lane["peak_equity_jpy"]), float(account["equity_jpy"]))
    lane["peak_equity_jpy"] = peak
    drawdown = (peak - float(account["equity_jpy"])) / float(
        lane["virtual_capital_jpy"]
    )
    if drawdown < float(lane["max_drawdown_fraction"]):
        return
    broker = lane["broker"]
    for order_id in list(broker.orders):
        broker.cancel_order(order_id)
    for trade_id in list(broker.positions):
        broker.close_trade(trade_id)
    lane["active"] = False
    lane["stop_reason"] = "MAX_DRAWDOWN_KILL_SWITCH"


def run(config_path: Path, *, resume: bool, poll_seconds: float) -> Path:
    config = load_experiment(config_path)
    root = (
        REPO_ROOT
        / "research/data/paper_champion_challenger_v1"
        / config["experiment_id"]
    )
    if root.exists() and not resume:
        raise ExperimentConfigError("experiment exists; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    state_path = root / "coordinator_state.json"
    previous = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if resume and state_path.exists()
        else {}
    )
    if previous and previous.get("config_file_sha256") != config_sha:
        raise ExperimentConfigError("resume config hash mismatch")

    Bot = _load_lab_bot()
    lanes: list[dict[str, Any]] = []
    for row in config["lanes"]:
        lane_root = root / row["lane_id"]
        lane_root.mkdir(parents=True, exist_ok=True)
        broker = VirtualBroker(
            ledger_path=lane_root / "ledger.jsonl",
            balance_jpy=float(row["virtual_capital_jpy"]),
            slippage_pips=float(row["costs"]["slippage_pips_per_fill"]),
            financing_pips_per_day=float(row["costs"]["financing_pips_per_day"]),
        )
        snap_path = lane_root / "broker_snapshot.json"
        if resume and snap_path.exists():
            broker.restore(json.loads(snap_path.read_text(encoding="utf-8")))
        bot = Bot(broker, dict(row["bot_config"]))
        prior_lane = next(
            (
                value
                for value in previous.get("lanes", [])
                if value.get("lane_id") == row["lane_id"]
            ),
            {},
        )
        lanes.append(
            {
                **row,
                "broker": broker,
                "bot": bot,
                "active": bool(prior_lane.get("active", True)),
                "stop_reason": prior_lane.get("stop_reason"),
                "peak_equity_jpy": float(
                    prior_lane.get("peak_equity_jpy", row["virtual_capital_jpy"])
                ),
            }
        )

    client = OandaReadOnlyClient()
    seed = seed_bots(client, [lane["bot"] for lane in lanes], config["pairs"], 1500)
    feed_chain = str(previous.get("feed_chain_sha256", "0" * 64))
    batch_count = int(previous.get("batch_count", 0))
    live_bars: dict[str, dict[str, Any]] = {}
    deadline = datetime.fromisoformat(
        config["window_end_utc"].replace("Z", "+00:00")
    ).timestamp()
    coordinator = {
        "contract": "QR_SHARED_CAUSAL_PAPER_RUNTIME_V1",
        "experiment_id": config["experiment_id"],
        "config_file_sha256": config_sha,
        "candidate_hash": config["candidate"]["candidate_hash"],
        "seed_receipt": seed,
        "feed_chain_sha256": feed_chain,
        "batch_count": batch_count,
        "status": "RUNNING",
        "authority": AUTHORITY,
    }
    _persist(root, lanes, coordinator)
    while time.time() < deadline:
        now = datetime.now(UTC)
        if not compute_market_status(now).is_fx_open:
            coordinator["status"] = "MARKET_CLOSED_WAIT"
            coordinator["updated_at_utc"] = now.isoformat()
            _persist(root, lanes, coordinator)
            time.sleep(30)
            continue
        fanout_started = False
        try:
            quotes = client.quotes(config["pairs"])
            if set(quotes) != set(config["pairs"]):
                raise VirtualBrokerError("incomplete shared quote batch")
            quote_batch = []
            canonical_quotes = []
            for pair in config["pairs"]:
                quote = quotes[pair]
                age = (now - quote.timestamp_utc).total_seconds()
                if age < -1 or age > 15:
                    raise VirtualBrokerError(f"stale quote: {pair}:{age:.3f}s")
                stamp = quote.timestamp_utc.isoformat()
                quote_batch.append((pair, quote.bid, quote.ask, stamp))
                canonical_quotes.append(
                    {"pair": pair, "bid": quote.bid, "ask": quote.ask, "ts": stamp}
                )
            batch_body = {
                "batch_index": batch_count,
                "previous_sha256": feed_chain,
                "quotes": canonical_quotes,
            }
            feed_chain = canonical_sha256(batch_body)
            fanout_started = True
            for lane in lanes:
                if lane["active"]:
                    lane["broker"].on_quote_batch(list(quote_batch))
            for pair in config["pairs"]:
                quote = quotes[pair]
                minute = int(quote.timestamp_utc.timestamp() // 60) * 60
                bar = live_bars.get(pair)
                if bar is not None and bar["epoch"] != minute:
                    for lane in lanes:
                        if lane["active"]:
                            lane["bot"].on_bar_closed(pair, dict(bar), bar["epoch"])
                    bar = None
                if bar is None:
                    live_bars[pair] = {
                        "epoch": minute,
                        "bid_o": quote.bid,
                        "bid_h": quote.bid,
                        "bid_l": quote.bid,
                        "bid_c": quote.bid,
                        "ask_o": quote.ask,
                        "ask_h": quote.ask,
                        "ask_l": quote.ask,
                        "ask_c": quote.ask,
                    }
                else:
                    bar["bid_h"] = max(bar["bid_h"], quote.bid)
                    bar["bid_l"] = min(bar["bid_l"], quote.bid)
                    bar["bid_c"] = quote.bid
                    bar["ask_h"] = max(bar["ask_h"], quote.ask)
                    bar["ask_l"] = min(bar["ask_l"], quote.ask)
                    bar["ask_c"] = quote.ask
            for lane in lanes:
                _kill_if_needed(lane)
            batch_count += 1
            coordinator.update(
                {
                    "status": "RUNNING",
                    "updated_at_utc": now.isoformat(),
                    "feed_chain_sha256": feed_chain,
                    "batch_count": batch_count,
                }
            )
            _persist(root, lanes, coordinator)
        except Exception as exc:
            coordinator.update(
                {
                    "status": (
                        "RUNTIME_ERROR_FAIL_CLOSED"
                        if fanout_started
                        else "FEED_ERROR_FAIL_CLOSED"
                    ),
                    "updated_at_utc": now.isoformat(),
                    "error": str(exc)[:300],
                }
            )
            _persist(root, lanes, coordinator)
            if fanout_started:
                raise
        time.sleep(poll_seconds)
    coordinator["status"] = "WINDOW_COMPLETE"
    coordinator["updated_at_utc"] = datetime.now(UTC).isoformat()
    _persist(root, lanes, coordinator)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_experiment(args.config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "experiment_id": config["experiment_id"],
                    "lane_count": len(config["lanes"]),
                    "authority": config["authority"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    root = run(args.config.resolve(), resume=args.resume, poll_seconds=args.poll_seconds)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
