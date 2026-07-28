from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .config import CryptoSafetyContract
from .ledger import CryptoLedger
from .paper import PaperEngine
from .stream import BitbankPublicStream

BPS = Decimal("10000")


def _d(value: object, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


@dataclass(frozen=True)
class FastPaperConfig:
    warmup_events: int = 12
    price_window: int = 32
    book_levels: int = 8
    max_data_age_ms: int = 3_000
    max_spread_bps: Decimal = Decimal("8")
    min_momentum_bps: Decimal = Decimal("0.02")
    min_imbalance: Decimal = Decimal("0.08")
    imbalance_edge_scale_bps: Decimal = Decimal("0.25")
    safety_buffer_bps: Decimal = Decimal("0.01")
    adverse_selection_bps: Decimal = Decimal("0.01")
    target_notional_jpy: Decimal = Decimal("5000")
    cooldown_ms: int = 200
    max_hold_ms: int = 8_000
    take_profit_bps: Decimal = Decimal("4")
    stop_loss_bps: Decimal = Decimal("3")
    telemetry_every_events: int = 250

    @classmethod
    def from_env(cls) -> "FastPaperConfig":
        return cls(
            warmup_events=int(
                os.environ.get("QR_CRYPTO_FAST_WARMUP_EVENTS", cls.warmup_events)
            ),
            price_window=int(
                os.environ.get("QR_CRYPTO_FAST_PRICE_WINDOW", cls.price_window)
            ),
            book_levels=int(
                os.environ.get("QR_CRYPTO_FAST_BOOK_LEVELS", cls.book_levels)
            ),
            max_data_age_ms=int(
                os.environ.get(
                    "QR_CRYPTO_FAST_MAX_DATA_AGE_MS", cls.max_data_age_ms
                )
            ),
            max_spread_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_MAX_SPREAD_BPS", cls.max_spread_bps
                )
            ),
            min_momentum_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_MIN_MOMENTUM_BPS", cls.min_momentum_bps
                )
            ),
            min_imbalance=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_MIN_IMBALANCE", cls.min_imbalance
                )
            ),
            imbalance_edge_scale_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_IMBALANCE_EDGE_SCALE_BPS",
                    cls.imbalance_edge_scale_bps,
                )
            ),
            safety_buffer_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_SAFETY_BUFFER_BPS", cls.safety_buffer_bps
                )
            ),
            adverse_selection_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_ADVERSE_SELECTION_BPS",
                    cls.adverse_selection_bps,
                )
            ),
            target_notional_jpy=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_TARGET_NOTIONAL_JPY",
                    cls.target_notional_jpy,
                )
            ),
            cooldown_ms=int(
                os.environ.get("QR_CRYPTO_FAST_COOLDOWN_MS", cls.cooldown_ms)
            ),
            max_hold_ms=int(
                os.environ.get("QR_CRYPTO_FAST_MAX_HOLD_MS", cls.max_hold_ms)
            ),
            take_profit_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_TAKE_PROFIT_BPS", cls.take_profit_bps
                )
            ),
            stop_loss_bps=_d(
                os.environ.get(
                    "QR_CRYPTO_FAST_STOP_LOSS_BPS", cls.stop_loss_bps
                )
            ),
            telemetry_every_events=int(
                os.environ.get(
                    "QR_CRYPTO_FAST_TELEMETRY_EVERY_EVENTS",
                    cls.telemetry_every_events,
                )
            ),
        )


@dataclass
class LocalOrderBook:
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    sequence: int | None = None
    whole_sequence: int | None = None
    published_at_ms: int = 0
    buffered_diffs: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    rejected_updates: int = 0

    @property
    def ready(self) -> bool:
        return self.sequence is not None and bool(self.bids) and bool(self.asks)

    def apply_whole(self, data: dict[str, Any]) -> bool:
        sequence = int(data.get("sequenceId", -1))
        bids = self._levels(data.get("bids", []))
        asks = self._levels(data.get("asks", []))
        if (
            sequence < 0
            or not bids
            or not asks
            or (
                self.whole_sequence is not None
                and sequence < self.whole_sequence
            )
        ):
            self.rejected_updates += 1
            return False
        self.bids = bids
        self.asks = asks
        self.sequence = sequence
        self.whole_sequence = sequence
        self.published_at_ms = int(data.get("timestamp", 0))
        buffered = sorted(
            (
                item
                for item in self.buffered_diffs
                if item[0] > sequence
            ),
            key=lambda item: item[0],
        )
        self.buffered_diffs = buffered
        for diff_sequence, diff in buffered:
            self._apply_diff_levels(diff_sequence, diff)
        return self._valid()

    def apply_diff(self, data: dict[str, Any]) -> bool:
        sequence = int(data.get("s", -1))
        if sequence < 0:
            self.rejected_updates += 1
            return False
        if len(self.buffered_diffs) >= 10_000:
            self.buffered_diffs.pop(0)
            self.rejected_updates += 1
        self.buffered_diffs.append((sequence, dict(data)))
        if self.sequence is None:
            return False
        if sequence <= self.sequence:
            return False
        self._apply_diff_levels(sequence, data)
        return self._valid()

    def _apply_diff_levels(self, sequence: int, data: dict[str, Any]) -> None:
        for raw_price, raw_amount in data.get("b", []):
            self._set_level(self.bids, raw_price, raw_amount)
        for raw_price, raw_amount in data.get("a", []):
            self._set_level(self.asks, raw_price, raw_amount)
        self.sequence = sequence
        self.published_at_ms = max(
            self.published_at_ms, int(data.get("t", 0))
        )

    @staticmethod
    def _levels(raw_levels: list[list[Any]]) -> dict[Decimal, Decimal]:
        result: dict[Decimal, Decimal] = {}
        for raw_price, raw_amount in raw_levels:
            price = _d(raw_price)
            amount = _d(raw_amount)
            if price > 0 and amount > 0:
                result[price] = amount
        return result

    @staticmethod
    def _set_level(
        side: dict[Decimal, Decimal], raw_price: object, raw_amount: object
    ) -> None:
        price = _d(raw_price)
        amount = _d(raw_amount)
        if price <= 0:
            return
        if amount <= 0:
            side.pop(price, None)
        else:
            side[price] = amount

    def _valid(self) -> bool:
        if not self.bids or not self.asks:
            self.rejected_updates += 1
            return False
        if max(self.bids) >= min(self.asks):
            self.rejected_updates += 1
            return False
        return True

    def depth(self) -> dict[str, list[list[str]]]:
        return {
            "bids": [
                [str(price), str(amount)]
                for price, amount in sorted(
                    self.bids.items(), reverse=True
                )
            ],
            "asks": [
                [str(price), str(amount)]
                for price, amount in sorted(self.asks.items())
            ],
        }

    def features(self, levels: int) -> dict[str, Decimal]:
        bids = sorted(self.bids.items(), reverse=True)[:levels]
        asks = sorted(self.asks.items())[:levels]
        if not bids or not asks:
            return {}
        bid = bids[0][0]
        ask = asks[0][0]
        mid = (bid + ask) / Decimal("2")
        bid_notional = sum((p * a for p, a in bids), Decimal("0"))
        ask_notional = sum((p * a for p, a in asks), Decimal("0"))
        total = bid_notional + ask_notional
        imbalance = (
            (bid_notional - ask_notional) / total
            if total > 0
            else Decimal("0")
        )
        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": (ask - bid) / mid * BPS,
            "imbalance": imbalance,
        }


@dataclass
class FastMarketState:
    pair: str
    price_window: int
    book: LocalOrderBook = field(default_factory=LocalOrderBook)
    prices: deque[Decimal] = field(init=False)
    event_count: int = 0
    ticker_at_ms: int = 0
    last_trade_at_ms: int = 0

    def __post_init__(self) -> None:
        self.prices = deque(maxlen=self.price_window)

    def observe_price(self, price: Decimal) -> None:
        if price > 0:
            self.prices.append(price)


class FastMicrostructureRouter:
    """Deterministic event router for spot and margin Paper intents."""

    def __init__(self, config: FastPaperConfig) -> None:
        self.config = config
        self.last_intent_ns: dict[str, int] = {}
        self.opened_ns: dict[str, int] = {}

    def decide(
        self,
        state: FastMarketState,
        *,
        position: Decimal,
        average_cost: Decimal,
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
        allow_short: bool,
        now_ns: int,
        wall_time_ms: int,
    ) -> dict[str, Any]:
        features = state.book.features(self.config.book_levels)
        if not features:
            return self._wait(state.pair, "BOOK_NOT_READY")
        age_ms = wall_time_ms - state.book.published_at_ms
        if age_ms < -1_000:
            return self._wait(state.pair, "FUTURE_STREAM_DATA")
        if age_ms > self.config.max_data_age_ms:
            return self._wait(state.pair, "STALE_STREAM_DATA")
        if state.event_count < self.config.warmup_events:
            return self._wait(state.pair, "WARMUP")
        if len(state.prices) < 3:
            return self._wait(state.pair, "PRICE_HISTORY_SHORT")
        last_intent = self.last_intent_ns.get(state.pair, 0)
        if now_ns - last_intent < self.config.cooldown_ms * 1_000_000:
            return self._wait(state.pair, "COOLDOWN")
        first = state.prices[0]
        last = state.prices[-1]
        momentum_bps = (
            (last - first) / first * BPS if first > 0 else Decimal("0")
        )
        spread_bps = features["spread_bps"]
        imbalance = features["imbalance"]
        maker_cost_bps = max(Decimal("0"), maker_fee_rate * BPS)
        taker_cost_bps = max(Decimal("0"), taker_fee_rate * BPS)
        expected_cost_bps = (
            maker_cost_bps
            + taker_cost_bps
            + spread_bps
            + self.config.adverse_selection_bps
        )
        order_flow_edge_bps = max(
            Decimal("0"),
            abs(imbalance) * self.config.imbalance_edge_scale_bps,
        )
        gross_edge_bps = max(
            Decimal("0"), abs(momentum_bps)
        ) + order_flow_edge_bps
        net_edge_bps = gross_edge_bps - expected_cost_bps
        common = {
            "pair": state.pair,
            "momentum_bps": str(momentum_bps),
            "spread_bps": str(spread_bps),
            "imbalance": str(imbalance),
            "maker_cost_bps": str(maker_cost_bps),
            "taker_cost_bps": str(taker_cost_bps),
            "round_trip_spread_bps": str(spread_bps),
            "expected_cost_bps": str(expected_cost_bps),
            "order_flow_edge_bps": str(order_flow_edge_bps),
            "gross_edge_bps": str(gross_edge_bps),
            "net_edge_bps": str(net_edge_bps),
            "book_sequence": state.book.sequence,
            "authority": "NONE",
            "live_permission": False,
        }
        if spread_bps > self.config.max_spread_bps:
            return {**common, "action": "WAIT", "reason": "WIDE_SPREAD"}
        if position == 0:
            if abs(imbalance) < self.config.min_imbalance:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "IMBALANCE_BELOW_ENTRY",
                }
            position_side = "LONG" if imbalance > 0 else "SHORT"
            if position_side == "SHORT" and not allow_short:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "SHORT_DISABLED",
                }
            adverse_momentum = (
                momentum_bps < -self.config.min_momentum_bps
                if position_side == "LONG"
                else momentum_bps > self.config.min_momentum_bps
            )
            if adverse_momentum:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "ADVERSE_MOMENTUM_BLOCK",
                }
            if net_edge_bps <= self.config.safety_buffer_bps:
                return {
                    **common,
                    "action": "WAIT",
                    "reason": "NET_EDGE_BELOW_BUFFER",
                }
            self.last_intent_ns[state.pair] = now_ns
            self.opened_ns[state.pair] = now_ns
            return {
                **common,
                "action": "ENTER",
                "position_side": position_side,
                "reason": f"MICROSTRUCTURE_{position_side}",
            }
        pnl_bps = (
            (features["bid"] - average_cost) / average_cost * BPS
            if position > 0 and average_cost > 0
            else (average_cost - features["ask"]) / average_cost * BPS
            if position < 0 and average_cost > 0
            else Decimal("0")
        )
        opened_ns = self.opened_ns.setdefault(state.pair, now_ns)
        held_ms = (now_ns - opened_ns) // 1_000_000
        exit_reason = None
        if pnl_bps >= self.config.take_profit_bps:
            exit_reason = "TAKE_PROFIT"
        elif pnl_bps <= -self.config.stop_loss_bps:
            exit_reason = "STOP_LOSS"
        elif held_ms >= self.config.max_hold_ms:
            exit_reason = "MAX_HOLD"
        elif (
            position > 0
            and momentum_bps <= -self.config.min_momentum_bps
        ) or (
            position < 0
            and momentum_bps >= self.config.min_momentum_bps
        ):
            exit_reason = "MOMENTUM_REVERSAL"
        elif (
            position > 0 and imbalance <= -self.config.min_imbalance
        ) or (
            position < 0 and imbalance >= self.config.min_imbalance
        ):
            exit_reason = "IMBALANCE_REVERSAL"
        if exit_reason:
            self.last_intent_ns[state.pair] = now_ns
            return {
                **common,
                "action": "EXIT",
                "position_side": "LONG" if position > 0 else "SHORT",
                "reason": exit_reason,
                "position_pnl_bps": str(pnl_bps),
                "held_ms": int(held_ms),
            }
        return {
            **common,
            "action": "WAIT",
            "position_side": "LONG" if position > 0 else "SHORT",
            "reason": "HOLD",
            "position_pnl_bps": str(pnl_bps),
            "held_ms": int(held_ms),
        }

    @staticmethod
    def _wait(pair: str, reason: str) -> dict[str, Any]:
        return {
            "pair": pair,
            "action": "WAIT",
            "reason": reason,
            "authority": "NONE",
            "live_permission": False,
        }


class FastPaperRunner:
    def __init__(
        self,
        ledger: CryptoLedger,
        paper: PaperEngine,
        *,
        stream: BitbankPublicStream | None = None,
        config: FastPaperConfig | None = None,
    ) -> None:
        self.safety = CryptoSafetyContract.from_env()
        self.safety.assert_safe()
        self.ledger = ledger
        self.paper = paper
        self.stream = stream or BitbankPublicStream()
        self.config = config or FastPaperConfig.from_env()
        self.router = FastMicrostructureRouter(self.config)

    async def run(
        self,
        pairs: list[str],
        pair_fees: dict[str, tuple[Decimal, Decimal]],
        *,
        duration_sec: float,
        max_events: int,
        daily_interest_rates: (
            dict[str, tuple[Decimal, Decimal]] | None
        ) = None,
        progress_callback: (
            Callable[[dict[str, Any]], object] | None
        ) = None,
        progress_interval_sec: float = 5.0,
    ) -> dict[str, Any]:
        normalized = [pair.lower() for pair in pairs]
        states = {
            pair: FastMarketState(pair, self.config.price_window)
            for pair in normalized
        }
        rooms = [
            f"{kind}_{pair}"
            for pair in normalized
            for kind in (
                "ticker",
                "transactions",
                "depth_whole",
                "depth_diff",
            )
        ]
        started_ns = time.monotonic_ns()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        run_id = hashlib.sha256(
            f"{time.time_ns()}|{','.join(normalized)}".encode()
        ).hexdigest()[:20]
        decision_latencies_us: list[float] = []
        exchange_latencies_ms: list[float] = []
        reasons: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        room_counts: Counter[str] = Counter()
        fills: list[dict[str, Any]] = []
        margin_call_recorded = False
        processed = 0
        next_progress_ns = started_ns
        progress_write_failures = 0

        def _emit_progress(status: str) -> None:
            nonlocal next_progress_ns, progress_write_failures
            if progress_callback is None:
                return
            now_ns = time.monotonic_ns()
            bids_now, asks_now = self._quotes(states)
            books_ready_now = sum(
                state.book.ready for state in states.values()
            )
            payload = {
                "schema": "QR_CRYPTO_PAPER_SHADOW_STATE_V1",
                "status": status,
                "run_id": run_id,
                "started_at_utc": started_at_utc,
                "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": (
                    "MARGIN" if self.paper.allow_short else "SPOT"
                ),
                "pairs": normalized,
                "events_processed": processed,
                "actions": dict(actions),
                "reasons": dict(reasons),
                "fills": len(fills),
                "books_ready": books_ready_now,
                "guardian": {
                    "state": (
                        "GREEN"
                        if books_ready_now == len(states) and processed > 0
                        else "RESTRICT"
                    ),
                    "kill_switch": False,
                    "deterministic": True,
                },
                "metrics": self.paper.mark_to_market(
                    bids_now, asks_now
                ),
                "safety": self.safety.as_dict(),
                "progress_write_failures": progress_write_failures,
            }
            try:
                progress_callback(payload)
            except Exception:
                progress_write_failures += 1
            next_progress_ns = now_ns + int(
                max(0.1, progress_interval_sec) * 1_000_000_000
            )

        _emit_progress("STARTING")

        async def _consume() -> None:
            nonlocal processed, margin_call_recorded
            async for message in self.stream.messages(
                rooms, max_messages=max_events
            ):
                wall_ms = int(time.time() * 1000)
                room = str(message.get("room_name", ""))
                pair = self._room_pair(room, normalized)
                if pair is None:
                    continue
                data = message.get("message", {}).get("data", {})
                if not isinstance(data, dict):
                    continue
                state = states[pair]
                updated, exchange_ms = self._apply_message(state, room, data)
                if not updated:
                    continue
                processed += 1
                room_counts[room[: -(len(pair) + 1)]] += 1
                state.event_count += 1
                if exchange_ms > 0:
                    exchange_latencies_ms.append(max(0.0, wall_ms - exchange_ms))
                features = state.book.features(self.config.book_levels)
                if features:
                    state.observe_price(features["mid"])
                maker_fee, taker_fee = pair_fees[pair]
                decision_started = time.monotonic_ns()
                position = self.paper.state.positions.get(pair, Decimal("0"))
                average_cost = self.paper.state.average_costs.get(
                    pair, Decimal("0")
                )
                decision = self.router.decide(
                    state,
                    position=position,
                    average_cost=average_cost,
                    maker_fee_rate=maker_fee,
                    taker_fee_rate=taker_fee,
                    allow_short=self.paper.allow_short,
                    now_ns=decision_started,
                    wall_time_ms=wall_ms,
                )
                decision_latencies_us.append(
                    (time.monotonic_ns() - decision_started) / 1_000
                )
                action = str(decision["action"])
                reason = str(decision["reason"])
                actions[action] += 1
                reasons[reason] += 1
                should_record = (
                    action != "WAIT"
                    or processed % self.config.telemetry_every_events == 0
                )
                if should_record:
                    self._record_decision(
                        decision,
                        pair=pair,
                        event_number=processed,
                        run_id=run_id,
                    )
                if action in {"ENTER", "EXIT"}:
                    intent = self._intent(
                        decision,
                        state,
                        position=position,
                        event_number=processed,
                        run_id=run_id,
                    )
                    fill = self.paper.process_intent(
                        intent,
                        depth=state.book.depth(),
                        maker_fee_rate=maker_fee,
                        taker_fee_rate=taker_fee,
                    )
                    fills.append(fill)
                if self.paper.allow_short:
                    bids_now, asks_now = self._quotes(states)
                    margin = self.paper.margin_snapshot(bids_now, asks_now)
                    if (
                        margin["status"] == "MODELED_MARGIN_CALL"
                        and not margin_call_recorded
                    ):
                        self.paper.state.margin_calls += 1
                        margin_call_recorded = True
                        self.ledger.append(
                            "MARGIN_GUARD",
                            run_id,
                            {
                                "status": margin["status"],
                                "margin_ratio": str(margin["margin_ratio"]),
                                "authority": "NONE",
                                "live_permission": False,
                            },
                            dedupe_key=f"margin-guard:{run_id}:call",
                        )
                    if margin["status"] == "MODELED_LOSSCUT":
                        fills.extend(
                            self._force_liquidate(
                                states,
                                pair_fees,
                                run_id=run_id,
                                event_number=processed,
                            )
                        )
                if time.monotonic_ns() >= next_progress_ns:
                    _emit_progress("RUNNING")

        timed_out = False
        try:
            await asyncio.wait_for(_consume(), timeout=duration_sec)
        except TimeoutError:
            timed_out = True
        elapsed_sec = max(
            0.000001, (time.monotonic_ns() - started_ns) / 1_000_000_000
        )
        interest_cost = self.paper.accrue_interest(
            daily_interest_rates or {},
            elapsed_sec=elapsed_sec,
            cause_id=run_id,
        )
        bids, asks = self._quotes(states)
        metrics = self.paper.mark_to_market(bids, asks)
        books_ready = sum(state.book.ready for state in states.values())
        guardian_issues: list[str] = []
        if books_ready != len(states):
            guardian_issues.append("BOOK_NOT_READY")
        if processed == 0:
            guardian_issues.append("NO_STREAM_EVENTS")
        guardian_state = "HALT" if processed == 0 else (
            "RESTRICT" if guardian_issues else "GREEN"
        )
        result = {
            "schema": "QR_CRYPTO_FAST_PAPER_CANARY_V1",
            "mode": (
                "PUBLIC_STREAM_EVENT_DRIVEN_MARGIN_PAPER"
                if self.paper.allow_short
                else "PUBLIC_STREAM_EVENT_DRIVEN_SPOT_PAPER"
            ),
            "venue": "bitbank",
            "run_id": run_id,
            "pairs": normalized,
            "rooms": rooms,
            "safety": self.safety.as_dict(),
            "guardian": {
                "state": guardian_state,
                "issues": guardian_issues,
                "kill_switch": guardian_state == "HALT",
                "deterministic": True,
            },
            "runtime": {
                "requested_duration_sec": duration_sec,
                "elapsed_sec": elapsed_sec,
                "max_events": max_events,
                "timed_out": timed_out,
                "events_processed": processed,
                "events_per_sec": processed / elapsed_sec,
                "books_ready": books_ready,
                "room_event_counts": dict(room_counts),
                "modeled_interest_cost_jpy": str(interest_cost),
            },
            "latency": {
                "decision_us_p50": _percentile(decision_latencies_us, 0.50),
                "decision_us_p95": _percentile(decision_latencies_us, 0.95),
                "decision_us_p99": _percentile(decision_latencies_us, 0.99),
                "exchange_to_receive_ms_p50": _percentile(
                    exchange_latencies_ms, 0.50
                ),
                "exchange_to_receive_ms_p95": _percentile(
                    exchange_latencies_ms, 0.95
                ),
            },
            "decisions": {
                "actions": dict(actions),
                "reasons": dict(reasons),
            },
            "fills": fills,
            "metrics": metrics,
            "ledger_integrity": self.ledger.verify(),
        }
        _emit_progress("EPOCH_COMPLETE")
        return result

    def _quotes(
        self, states: dict[str, FastMarketState]
    ) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        bids: dict[str, Decimal] = {}
        asks: dict[str, Decimal] = {}
        for pair, state in states.items():
            features = state.book.features(self.config.book_levels)
            if features:
                bids[pair] = features["bid"]
                asks[pair] = features["ask"]
        return bids, asks

    def _force_liquidate(
        self,
        states: dict[str, FastMarketState],
        pair_fees: dict[str, tuple[Decimal, Decimal]],
        *,
        run_id: str,
        event_number: int,
    ) -> list[dict[str, Any]]:
        self.paper.state.forced_liquidations += 1
        self.ledger.append(
            "MARGIN_GUARD",
            run_id,
            {
                "status": "MODELED_LOSSCUT",
                "action": "FORCE_CLOSE_ALL_PAPER_POSITIONS",
                "authority": "NONE",
                "live_permission": False,
            },
            dedupe_key=f"margin-guard:{run_id}:losscut",
        )
        results: list[dict[str, Any]] = []
        for pair, amount in list(self.paper.state.positions.items()):
            if amount == 0 or not states[pair].book.ready:
                continue
            maker_fee, taker_fee = pair_fees[pair]
            position_side = "LONG" if amount > 0 else "SHORT"
            side = "SELL" if amount > 0 else "BUY"
            intent = {
                "intent_id": hashlib.sha256(
                    f"{run_id}|losscut|{pair}|{event_number}".encode()
                ).hexdigest()[:24],
                "pair": pair,
                "side": side,
                "position_effect": "CLOSE",
                "position_side": position_side,
                "amount": str(abs(amount)),
                "order_style": "PAPER_TAKER",
                "regime": "MODELED_LOSSCUT",
                "strategy": "FAST_MICROSTRUCTURE",
                "signal_reason": "MODELED_LOSSCUT",
                "run_id": run_id,
                "event_at_utc": datetime.now(timezone.utc).isoformat(),
                "guardian": "MODELED_LOSSCUT",
                "authority": "NONE",
                "live_permission": False,
            }
            results.append(
                self.paper.process_intent(
                    intent,
                    depth=states[pair].book.depth(),
                    maker_fee_rate=maker_fee,
                    taker_fee_rate=taker_fee,
                )
            )
        return results

    @staticmethod
    def _room_pair(room: str, pairs: list[str]) -> str | None:
        for pair in pairs:
            if room.endswith(f"_{pair}"):
                return pair
        return None

    @staticmethod
    def _apply_message(
        state: FastMarketState, room: str, data: dict[str, Any]
    ) -> tuple[bool, int]:
        if room.startswith("depth_whole_"):
            return state.book.apply_whole(data), int(data.get("timestamp", 0))
        if room.startswith("depth_diff_"):
            return state.book.apply_diff(data), int(data.get("t", 0))
        if room.startswith("ticker_"):
            buy = _d(data.get("buy"))
            sell = _d(data.get("sell"))
            timestamp = int(data.get("timestamp", 0))
            state.ticker_at_ms = timestamp
            if buy > 0 and sell > buy:
                state.observe_price((buy + sell) / Decimal("2"))
                return True, timestamp
            return False, timestamp
        if room.startswith("transactions_"):
            transactions = data.get("transactions", [])
            if not isinstance(transactions, list) or not transactions:
                return False, 0
            latest = max(
                transactions, key=lambda row: int(row.get("executed_at", 0))
            )
            timestamp = int(latest.get("executed_at", 0))
            state.last_trade_at_ms = timestamp
            state.observe_price(_d(latest.get("price")))
            return True, timestamp
        return False, 0

    def _record_decision(
        self,
        decision: dict[str, Any],
        *,
        pair: str,
        event_number: int,
        run_id: str,
    ) -> None:
        self.ledger.append(
            "FAST_DECISION",
            pair,
            decision,
            dedupe_key=f"fast-decision:{run_id}:{pair}:{event_number}",
        )

    def _intent(
        self,
        decision: dict[str, Any],
        state: FastMarketState,
        *,
        position: Decimal,
        event_number: int,
        run_id: str,
    ) -> dict[str, Any]:
        features = state.book.features(self.config.book_levels)
        position_side = str(decision["position_side"])
        opening = decision["action"] == "ENTER"
        side = (
            "BUY"
            if (opening and position_side == "LONG")
            or (not opening and position_side == "SHORT")
            else "SELL"
        )
        price = features["bid"] if side == "BUY" else features["ask"]
        amount = (
            self.config.target_notional_jpy / price
            if opening and price > 0
            else abs(position)
        )
        raw_id = (
            f"{run_id}|{state.pair}|{side}|{state.book.sequence}|{event_number}|"
            f"{decision['reason']}"
        )
        return {
            "intent_id": hashlib.sha256(raw_id.encode()).hexdigest()[:24],
            "pair": state.pair,
            "side": side,
            "position_effect": "OPEN" if opening else "CLOSE",
            "position_side": position_side,
            "amount": str(amount),
            "order_style": (
                "PAPER_MAKER_LIMIT" if opening else "PAPER_TAKER"
            ),
            "limit_price": str(price),
            "regime": "FAST_MICROSTRUCTURE",
            "strategy": "FAST_MICROSTRUCTURE",
            "signal_reason": decision["reason"],
            "run_id": run_id,
            "event_at_utc": datetime.now(timezone.utc).isoformat(),
            "guardian": "GREEN",
            "authority": "NONE",
            "live_permission": False,
        }


def fast_report_markdown(result: dict[str, Any]) -> str:
    runtime = result["runtime"]
    latency = result["latency"]
    metrics = result["metrics"]
    actions = result["decisions"]["actions"]
    top_reasons = sorted(
        result["decisions"]["reasons"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return "\n".join(
        [
            f"# QuantRabbit Crypto｜bitbank "
            f"{'Margin' if 'MARGIN' in result['mode'] else 'Spot'} "
            "Fast Paper Canary",
            "",
            f"- Mode: `{result['mode']}`",
            f"- Run ID: `{result['run_id']}`",
            f"- Initial cash JPY: `{metrics['initial_cash_jpy']}`",
            f"- Pairs: `{', '.join(result['pairs'])}`",
            f"- Guardian: `{result['guardian']['state']}`",
            f"- Events: `{runtime['events_processed']}`",
            f"- Events/sec: `{runtime['events_per_sec']:.3f}`",
            f"- Decision p50/p95/p99 μs: "
            f"`{latency['decision_us_p50']}` / "
            f"`{latency['decision_us_p95']}` / "
            f"`{latency['decision_us_p99']}`",
            f"- ENTER / EXIT / WAIT: `{actions.get('ENTER', 0)}` / "
            f"`{actions.get('EXIT', 0)}` / `{actions.get('WAIT', 0)}`",
            f"- Top reasons: `"
            f"{', '.join(f'{reason}={count}' for reason, count in top_reasons)}"
            f"`",
            f"- Fills: `{metrics['trade_count']}`",
            f"- Round trips: `{metrics['round_trip_count']}`",
            f"- Short positions remaining: "
            f"`{metrics['short_position_count']}`",
            f"- Gross exposure JPY: `{metrics['gross_exposure_jpy']}`",
            f"- Effective leverage: `{metrics['effective_leverage']}`",
            f"- Margin ratio: `{metrics['margin_ratio']}`",
            f"- Margin status: `{metrics['margin_status']}`",
            f"- Net PnL JPY: `{metrics['net_pnl_jpy']}`",
            f"- Max drawdown JPY: `{metrics['max_drawdown_jpy']}`",
            f"- Fees JPY: `{metrics['fees_jpy']}`",
            f"- Interest JPY: `{metrics['interest_cost_jpy']}`",
            "",
            "This is deterministic Public Stream Paper execution. "
            "It has no live order authority and does not guarantee profit.",
            "",
        ]
    )
