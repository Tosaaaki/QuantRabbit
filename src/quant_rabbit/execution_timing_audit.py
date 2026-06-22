"""Audit missed execution timing and late loss exits.

This module turns two operator questions into machine-readable metrics:

* Did a canceled pending entry later touch its entry / TP and move profitable?
* Did a losing close have a profitable MFE window before it was closed red?
* Did a market close leave same-side follow-through behind, or did it avoid
  adverse giveback after the close?

The first implementation uses OANDA M1 bid/ask candles. Tick JSONL can replace
the candle fetcher later without changing the output contract.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    TP_PROGRESS_REPAIR_REPLAY_FIELD,
)
from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_EXECUTION_TIMING_AUDIT_REPORT,
)


# The live decision cadence is roughly 20 minutes; six hours covers eighteen
# missed decision opportunities and matches the legacy "recovered in 6h" stop
# regret lens already mined into strategy reports. It is an audit window, not a
# live risk threshold.
DEFAULT_POST_CANCEL_HORIZON_HOURS = 6.0

# Use the same audit horizon as canceled pending regrets so close timing and
# cancel timing are compared over the same next-cycle opportunity window.
DEFAULT_POST_CLOSE_HORIZON_HOURS = 6.0

# The live-facing audit should compare the current operating week, not old
# May-era repairs, by default. Operators can widen this explicitly.
DEFAULT_LOOKBACK_HOURS = 168.0

# OANDA caps candle responses. Six-hour chunks keep M1 requests comfortably
# below that cap while preserving enough resolution for timing lag metrics.
DEFAULT_CANDLE_CHUNK_HOURS = 6.0

# Shape rollups are compact routing hints for the next cycle digest/GPT packet;
# individual rows remain in `canceled_order_regrets` for full audit detail.
TIMING_SHAPE_ROLLUP_LIMIT = 12

# Mirror the PositionManager TP-progress harvest contract for audit-only
# detection. This does not execute exits; it marks loss closes where the trade
# had already captured enough of its attached TP that the fast guardian should
# have had a profit-banking opportunity to evaluate.
TP_PROGRESS_CAPTURE_THRESHOLD = float(
    os.environ.get(
        "QR_TIMING_PROFIT_CAPTURE_MIN_TP_PROGRESS",
        os.environ.get("QR_TP_PROGRESS_PROFIT_TAKE_MIN_PROGRESS", "0.30"),
    )
)

# Mirror the live PositionManager TP-progress market-close noise floor. The
# audit still uses candles, not ticks, so the ATR/spread values are conservative
# bar-level approximations, but the replay contract now matches the production
# gate shape instead of a raw MFE threshold.
TP_PROGRESS_REPAIR_NOISE_MULT = float(
    os.environ.get("QR_TEMPORARY_EXTREME_MIN_PROFIT_NOISE_MULT", "1.0")
)
TP_PROGRESS_REPAIR_ATR_PERIOD = int(
    os.environ.get("QR_TIMING_PROFIT_CAPTURE_ATR_PERIOD", "14")
)


@dataclass(frozen=True)
class BidAskCandle:
    timestamp_utc: datetime
    bid_high: float
    bid_low: float
    ask_high: float
    ask_low: float


@dataclass(frozen=True)
class _CanceledOrder:
    order_id: str
    accepted_at_utc: datetime
    canceled_at_utc: datetime
    pair: str
    side: str
    order_type: str
    units: int
    entry: float
    tp: float | None
    sl: float | None
    lane_id: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class _MarketClose:
    trade_id: str
    fill_at_utc: datetime
    close_at_utc: datetime
    pair: str
    side: str
    units: int
    entry: float
    close_price: float
    realized_pl_jpy: float
    exit_reason: str | None
    gateway_action: str | None
    lane_id: str | None
    tp: float | None
    sl: float | None
    close_raw: dict[str, Any]


CandleFetcher = Callable[[str, datetime, datetime, str], tuple[BidAskCandle, ...]]


def build_execution_timing_audit(
    *,
    ledger_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
    snapshot_path: Path | None = DEFAULT_BROKER_SNAPSHOT,
    output_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
    report_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT_REPORT,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    post_cancel_hours: float = DEFAULT_POST_CANCEL_HORIZON_HOURS,
    post_close_hours: float = DEFAULT_POST_CLOSE_HORIZON_HOURS,
    granularity: str = "M1",
    now_utc: datetime | None = None,
    candle_fetcher: CandleFetcher | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Build and persist the execution timing audit payload."""

    now = _normalize_utc(now_utc or datetime.now(timezone.utc))
    lookback_delta = timedelta(hours=float(lookback_hours))
    cutoff = now - lookback_delta
    home_conversions = _home_conversions_from_snapshot(snapshot_path)
    canceled, loss_closes, market_closes, market_close_cutoff, market_close_anchor = _load_candidates(
        ledger_path=ledger_path,
        cutoff=cutoff,
        lookback_delta=lookback_delta,
        max_events=max_events,
    )
    windows = _candidate_windows(
        canceled,
        loss_closes,
        market_closes,
        post_cancel_hours=float(post_cancel_hours),
        post_close_hours=float(post_close_hours),
        now_utc=now,
    )
    candle_cache, fetch_errors = _load_candle_cache(
        windows,
        granularity=granularity,
        candle_fetcher=candle_fetcher,
    )
    granularity_delta = _granularity_delta(granularity)
    canceled_rows = [
        _audit_canceled_order(
            order,
            _candles_between(
                candle_cache.get(order.pair, ()),
                _first_complete_candle_start_at_or_after(order.canceled_at_utc, granularity_delta),
                min(order.canceled_at_utc + timedelta(hours=float(post_cancel_hours)), now),
            ),
            home_conversions,
        )
        for order in canceled
    ]
    loss_rows = [
        _audit_loss_close(
            close,
            _candles_between(
                candle_cache.get(close.pair, ()),
                _first_complete_candle_start_at_or_after(close.fill_at_utc, granularity_delta),
                _last_complete_candle_start_before(close.close_at_utc, granularity_delta),
            ),
            home_conversions,
        )
        for close in loss_closes
    ]
    market_close_rows = [
        _audit_market_close_counterfactual(
            close,
            _candles_between(
                candle_cache.get(close.pair, ()),
                _first_complete_candle_start_at_or_after(close.close_at_utc, granularity_delta),
                min(close.close_at_utc + timedelta(hours=float(post_close_hours)), now),
            ),
            home_conversions,
        )
        for close in market_closes
    ]
    payload = {
        "generated_at_utc": now.isoformat(),
        "status": "PARTIAL_DATA" if fetch_errors else "OK",
        "precision": {
            "price_basis": "OANDA_M1_BID_ASK_CANDLES",
            "granularity": granularity,
            "note": "M1 candles prove bar-level opportunity; tick JSONL can refine ordering inside a candle.",
            TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
        },
        "window": {
            "from_utc": min(cutoff, market_close_cutoff).isoformat(),
            "canceled_from_utc": cutoff.isoformat(),
            "market_close_from_utc": market_close_cutoff.isoformat(),
            "market_close_anchor_utc": market_close_anchor.isoformat() if market_close_anchor else None,
            "to_utc": now.isoformat(),
            "lookback_hours": float(lookback_hours),
            "post_cancel_hours": float(post_cancel_hours),
            "post_close_hours": float(post_close_hours),
        },
        "summary": _summary(canceled_rows, loss_rows, market_close_rows),
        "canceled_order_regret_by_shape": _canceled_order_regret_by_shape(canceled_rows),
        "canceled_order_regrets": canceled_rows,
        "loss_close_regrets": loss_rows,
        "market_close_counterfactuals": market_close_rows,
        "fetch_errors": fetch_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _write_report(payload, report_path)
    return payload


def fetch_bid_ask_candles_between(
    pair: str,
    time_from: datetime,
    time_to: datetime,
    granularity: str,
    *,
    client: OandaReadOnlyClient | None = None,
) -> tuple[BidAskCandle, ...]:
    client = client or OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {
            "granularity": granularity,
            "from": _format_oanda_time(time_from),
            "to": _format_oanda_time(time_to),
            "price": "BA",
            "includeFirst": "true",
        },
    )
    candles: list[BidAskCandle] = []
    for item in payload.get("candles") or []:
        bid = item.get("bid") if isinstance(item.get("bid"), dict) else {}
        ask = item.get("ask") if isinstance(item.get("ask"), dict) else {}
        try:
            ts = _parse_utc(item.get("time"))
            if ts is None:
                continue
            candles.append(
                BidAskCandle(
                    timestamp_utc=ts,
                    bid_high=float(bid.get("h")),
                    bid_low=float(bid.get("l")),
                    ask_high=float(ask.get("h")),
                    ask_low=float(ask.get("l")),
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(candles)


def _load_candidates(
    *,
    ledger_path: Path,
    cutoff: datetime,
    lookback_delta: timedelta,
    max_events: int | None,
) -> tuple[list[_CanceledOrder], list[_MarketClose], list[_MarketClose], datetime, datetime | None]:
    if not ledger_path.exists():
        return [], [], [], cutoff, None
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        accepted = [_row_dict(row) for row in conn.execute("SELECT * FROM execution_events WHERE event_type='ORDER_ACCEPTED'")]
        canceled = [_row_dict(row) for row in conn.execute("SELECT * FROM execution_events WHERE event_type='ORDER_CANCELED'")]
        filled = [_row_dict(row) for row in conn.execute("SELECT * FROM execution_events WHERE event_type='ORDER_FILLED'")]
        closed = [_row_dict(row) for row in conn.execute("SELECT * FROM execution_events WHERE event_type='TRADE_CLOSED'")]
        protections = [_row_dict(row) for row in conn.execute("SELECT * FROM execution_events WHERE event_type='PROTECTION_CREATED'")]
        gateway_closes = [
            _row_dict(row)
            for row in conn.execute("SELECT * FROM execution_events WHERE event_type='GATEWAY_TRADE_CLOSE_SENT'")
        ]
    market_close_anchor = _latest_market_close_timestamp(closed)
    market_close_cutoff = cutoff
    if market_close_anchor is not None:
        # Profitability acceptance judges leakage over the trading week ending
        # at the latest market close. Match that anchor here so older closes in
        # the same acceptance window do not become UNCLASSIFIED timing leaks
        # after a quiet weekend or paused market.
        market_close_cutoff = min(cutoff, market_close_anchor - lookback_delta)
    cancels_by_order: dict[str, dict[str, Any]] = {}
    for row in canceled:
        oid = str(row.get("order_id") or "")
        ts = _parse_utc(row.get("ts_utc"))
        if not oid or ts is None:
            continue
        prev = cancels_by_order.get(oid)
        if prev is None or ts < _parse_utc(prev.get("ts_utc")):
            cancels_by_order[oid] = row
    fill_times_by_order: dict[str, datetime] = {}
    fills_by_trade: dict[str, dict[str, Any]] = {}
    for row in filled:
        ts = _parse_utc(row.get("ts_utc"))
        if ts is None:
            continue
        oid = str(row.get("order_id") or "")
        if oid:
            current = fill_times_by_order.get(oid)
            if current is None or ts < current:
                fill_times_by_order[oid] = ts
        trade_id = str(row.get("trade_id") or "")
        if trade_id and (trade_id not in fills_by_trade or ts < _parse_utc(fills_by_trade[trade_id].get("ts_utc"))):
            fills_by_trade[trade_id] = row
    canceled_candidates: list[_CanceledOrder] = []
    for row in accepted:
        if str(row.get("exit_reason") or "").upper() != "CLIENT_ORDER":
            continue
        oid = str(row.get("order_id") or "")
        cancel = cancels_by_order.get(oid)
        if not oid or not cancel:
            continue
        accepted_at = _parse_utc(row.get("ts_utc"))
        canceled_at = _parse_utc(cancel.get("ts_utc"))
        if accepted_at is None or canceled_at is None or canceled_at < cutoff:
            continue
        filled_at = fill_times_by_order.get(oid)
        if filled_at is not None and filled_at <= canceled_at:
            continue
        raw = _json_obj(row.get("raw_json"))
        order_type = str(raw.get("type") or "").upper()
        if order_type not in {"LIMIT_ORDER", "STOP_ORDER"}:
            continue
        pair = str(row.get("pair") or raw.get("instrument") or "").strip()
        side = str(row.get("side") or _side_from_units(raw.get("units")) or "").upper()
        entry = _float(row.get("price")) or _float(raw.get("price"))
        if not pair or side not in {"LONG", "SHORT"} or entry is None:
            continue
        units = abs(int(float(row.get("units") or raw.get("units") or 0)))
        canceled_candidates.append(
            _CanceledOrder(
                order_id=oid,
                accepted_at_utc=accepted_at,
                canceled_at_utc=canceled_at,
                pair=pair,
                side=side,
                order_type=order_type,
                units=units,
                entry=entry,
                tp=_float(row.get("tp")) or _nested_price(raw.get("takeProfitOnFill")),
                sl=_float(row.get("sl")) or _nested_price(raw.get("stopLossOnFill")),
                lane_id=str(row.get("lane_id") or "") or None,
                raw=raw,
            )
        )
    protections_by_trade = _protections_by_trade(protections)
    gateway_closes_by_trade = _gateway_closes_by_trade(gateway_closes)
    loss_close_candidates: list[_MarketClose] = []
    market_close_candidates: list[_MarketClose] = []
    for row in closed:
        close_at = _parse_utc(row.get("ts_utc"))
        if close_at is None or close_at < market_close_cutoff:
            continue
        exit_reason = str(row.get("exit_reason") or "").upper()
        realized = _float(row.get("realized_pl_jpy"))
        include_loss_close = realized is not None and realized < 0.0
        include_market_close = exit_reason == "MARKET_ORDER_TRADE_CLOSE"
        if not include_loss_close and not include_market_close:
            continue
        trade_id = str(row.get("trade_id") or "")
        fill = fills_by_trade.get(trade_id)
        if not trade_id or not fill:
            continue
        fill_at = _parse_utc(fill.get("ts_utc"))
        if fill_at is None or fill_at >= close_at:
            continue
        close_raw = _json_obj(row.get("raw_json"))
        pair = str(row.get("pair") or fill.get("pair") or close_raw.get("instrument") or "").strip()
        side = str(fill.get("side") or row.get("side") or "").upper()
        entry = _float(fill.get("price"))
        close_price = _float(row.get("price"))
        if not pair or side not in {"LONG", "SHORT"} or entry is None or close_price is None or realized is None:
            continue
        tp, sl = _latest_protection_before(protections_by_trade.get(trade_id, ()), close_at)
        gateway_action = _latest_gateway_close_action_before(gateway_closes_by_trade.get(trade_id, ()), close_at)
        candidate = _MarketClose(
            trade_id=trade_id,
            fill_at_utc=fill_at,
            close_at_utc=close_at,
            pair=pair,
            side=side,
            units=abs(int(float(fill.get("units") or row.get("units") or 0))),
            entry=entry,
            close_price=close_price,
            realized_pl_jpy=realized,
            exit_reason=str(row.get("exit_reason") or "") or None,
            gateway_action=gateway_action,
            lane_id=str(fill.get("lane_id") or row.get("lane_id") or "") or None,
            tp=tp,
            sl=sl,
            close_raw=close_raw,
        )
        if include_loss_close:
            loss_close_candidates.append(candidate)
        if include_market_close:
            market_close_candidates.append(candidate)
    canceled_candidates.sort(key=lambda item: item.canceled_at_utc, reverse=True)
    loss_close_candidates.sort(key=lambda item: item.close_at_utc, reverse=True)
    market_close_candidates.sort(key=lambda item: item.close_at_utc, reverse=True)
    if max_events is not None and max_events > 0:
        canceled_candidates = canceled_candidates[:max_events]
        loss_close_candidates = loss_close_candidates[:max_events]
        market_close_candidates = market_close_candidates[:max_events]
    return canceled_candidates, loss_close_candidates, market_close_candidates, market_close_cutoff, market_close_anchor


def _latest_market_close_timestamp(rows: Iterable[dict[str, Any]]) -> datetime | None:
    latest: datetime | None = None
    for row in rows:
        if str(row.get("exit_reason") or "").upper() != "MARKET_ORDER_TRADE_CLOSE":
            continue
        ts = _parse_utc(row.get("ts_utc"))
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    return latest


def _candidate_windows(
    canceled: Iterable[_CanceledOrder],
    loss_closes: Iterable[_MarketClose],
    market_closes: Iterable[_MarketClose],
    *,
    post_cancel_hours: float,
    post_close_hours: float,
    now_utc: datetime,
) -> dict[str, list[tuple[datetime, datetime]]]:
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    now = _normalize_utc(now_utc)
    for order in canceled:
        end = min(order.canceled_at_utc + timedelta(hours=post_cancel_hours), now)
        if end > order.canceled_at_utc:
            windows.setdefault(order.pair, []).append((order.canceled_at_utc, end))
    for close in loss_closes:
        windows.setdefault(close.pair, []).append((close.fill_at_utc, close.close_at_utc))
    for close in market_closes:
        end = min(close.close_at_utc + timedelta(hours=post_close_hours), now)
        if end > close.close_at_utc:
            windows.setdefault(close.pair, []).append((close.close_at_utc, end))
    return windows


def _load_candle_cache(
    windows: dict[str, list[tuple[datetime, datetime]]],
    *,
    granularity: str,
    candle_fetcher: CandleFetcher | None,
) -> tuple[dict[str, tuple[BidAskCandle, ...]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not windows:
        return {}, errors
    fetcher = candle_fetcher
    client: OandaReadOnlyClient | None = None
    if fetcher is None:
        try:
            client = OandaReadOnlyClient()
        except Exception as exc:
            return {}, [{"pair": "*", "error": f"{exc.__class__.__name__}: {exc}"}]

        def fetcher(pair: str, start: datetime, end: datetime, granularity: str) -> tuple[BidAskCandle, ...]:
            return fetch_bid_ask_candles_between(pair, start, end, granularity, client=client)

    cache: dict[str, list[BidAskCandle]] = {}
    for pair, intervals in windows.items():
        for start, end in _merge_intervals(intervals):
            for chunk_start, chunk_end in _chunk_interval(start, end):
                try:
                    cache.setdefault(pair, []).extend(fetcher(pair, chunk_start, chunk_end, granularity))
                except Exception as exc:
                    errors.append(
                        {
                            "pair": pair,
                            "from_utc": chunk_start.isoformat(),
                            "to_utc": chunk_end.isoformat(),
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
    normalized = {
        pair: tuple(sorted({_candle_key(c): c for c in candles}.values(), key=lambda c: c.timestamp_utc))
        for pair, candles in cache.items()
    }
    return normalized, errors


def _audit_canceled_order(
    order: _CanceledOrder,
    candles: tuple[BidAskCandle, ...],
    home_conversions: dict[str, float],
) -> dict[str, Any]:
    touched_at: datetime | None = None
    after_touch: list[BidAskCandle] = []
    for idx, candle in enumerate(candles):
        if _entry_touched(order.order_type, order.side, order.entry, candle):
            touched_at = candle.timestamp_utc
            after_touch = list(candles[idx:])
            break
    mfe_pips = 0.0
    mfe_at: datetime | None = None
    first_positive_at: datetime | None = None
    tp_touch_at: datetime | None = None
    sl_touch_at: datetime | None = None
    same_candle_entry_tp = False
    if touched_at is not None:
        for candle in after_touch:
            favorable = _favorable_delta(order.side, order.entry, candle)
            if favorable > 0 and first_positive_at is None:
                first_positive_at = candle.timestamp_utc
            if favorable > mfe_pips / _pip_factor(order.pair):
                mfe_pips = favorable * _pip_factor(order.pair)
                mfe_at = candle.timestamp_utc
            if order.tp is not None and tp_touch_at is None and _tp_touched(order.side, order.tp, candle):
                tp_touch_at = candle.timestamp_utc
                same_candle_entry_tp = candle.timestamp_utc == touched_at
            if order.sl is not None and sl_touch_at is None and _sl_touched(order.side, order.sl, candle):
                sl_touch_at = candle.timestamp_utc
    jpy_per_pip = _jpy_per_pip(order.pair, order.units, home_conversions=home_conversions, raw=None)
    missed_mfe_jpy = round(mfe_pips * jpy_per_pip, 4) if jpy_per_pip is not None else None
    return {
        "order_id": order.order_id,
        "lane_id": order.lane_id,
        "pair": order.pair,
        "side": order.side,
        "order_type": order.order_type,
        "units": order.units,
        "entry": order.entry,
        "tp": order.tp,
        "sl": order.sl,
        "accepted_at_utc": order.accepted_at_utc.isoformat(),
        "canceled_at_utc": order.canceled_at_utc.isoformat(),
        "entry_touched_after_cancel": touched_at is not None,
        "entry_touch_after_cancel_minutes": _minutes_between(order.canceled_at_utc, touched_at),
        "first_positive_after_cancel_minutes": _minutes_between(order.canceled_at_utc, first_positive_at),
        "tp_touched_after_cancel": tp_touch_at is not None,
        "tp_touch_after_cancel_minutes": _minutes_between(order.canceled_at_utc, tp_touch_at),
        "sl_touched_after_cancel": sl_touch_at is not None,
        "same_candle_entry_tp_ambiguous": same_candle_entry_tp,
        "mfe_pips_after_cancel_entry": round(mfe_pips, 4),
        "mfe_at_utc": mfe_at.isoformat() if mfe_at else None,
        "estimated_missed_mfe_jpy": missed_mfe_jpy,
        "jpy_per_pip_estimate": round(jpy_per_pip, 6) if jpy_per_pip is not None else None,
        "candles_available": len(candles),
    }


def _audit_loss_close(
    close: _MarketClose,
    candles: tuple[BidAskCandle, ...],
    home_conversions: dict[str, float],
) -> dict[str, Any]:
    mfe_pips = 0.0
    mfe_at: datetime | None = None
    first_positive_at: datetime | None = None
    tp_touch_at: datetime | None = None
    sl_touch_at: datetime | None = None
    for candle in candles:
        favorable = _favorable_delta(close.side, close.entry, candle)
        if favorable > 0 and first_positive_at is None:
            first_positive_at = candle.timestamp_utc
        if favorable > mfe_pips / _pip_factor(close.pair):
            mfe_pips = favorable * _pip_factor(close.pair)
            mfe_at = candle.timestamp_utc
        if close.tp is not None and tp_touch_at is None and _tp_touched(close.side, close.tp, candle):
            tp_touch_at = candle.timestamp_utc
        if close.sl is not None and sl_touch_at is None and _sl_touched(close.side, close.sl, candle):
            sl_touch_at = candle.timestamp_utc
    jpy_per_pip = _jpy_per_pip(
        close.pair,
        close.units,
        home_conversions=home_conversions,
        raw=close.close_raw,
    )
    estimated_mfe_jpy = round(mfe_pips * jpy_per_pip, 4) if jpy_per_pip is not None else None
    tp_distance_pips = _tp_distance_pips(close)
    tp_progress = round(mfe_pips / tp_distance_pips, 4) if tp_distance_pips and tp_distance_pips > 0 else None
    capture_missed = bool(first_positive_at is not None) and (
        tp_touch_at is not None
        or (tp_progress is not None and tp_progress >= TP_PROGRESS_CAPTURE_THRESHOLD)
    )
    counterfactual = _profit_capture_counterfactual(
        close,
        capture_missed=capture_missed,
        tp_touch_at=tp_touch_at,
        tp_distance_pips=tp_distance_pips,
        mfe_pips=mfe_pips,
        jpy_per_pip=jpy_per_pip,
    )
    repair_replay = _profit_capture_repair_replay(
        close,
        candles,
        tp_distance_pips=tp_distance_pips,
        jpy_per_pip=jpy_per_pip,
    )
    return {
        "trade_id": close.trade_id,
        "lane_id": close.lane_id,
        "pair": close.pair,
        "side": close.side,
        "units": close.units,
        "entry": close.entry,
        "close_price": close.close_price,
        "tp": close.tp,
        "sl": close.sl,
        "exit_reason": close.exit_reason,
        "gateway_action": close.gateway_action,
        "realized_pl_jpy": round(close.realized_pl_jpy, 4),
        "fill_at_utc": close.fill_at_utc.isoformat(),
        "close_at_utc": close.close_at_utc.isoformat(),
        "had_positive_mfe_before_loss_close": first_positive_at is not None,
        "first_positive_minutes_after_fill": _minutes_between(close.fill_at_utc, first_positive_at),
        "decision_lag_minutes_after_first_positive": _minutes_between(first_positive_at, close.close_at_utc),
        "tp_touched_before_loss_close": tp_touch_at is not None,
        "tp_touch_minutes_after_fill": _minutes_between(close.fill_at_utc, tp_touch_at),
        "sl_touched_before_loss_close": sl_touch_at is not None,
        "mfe_pips_before_loss_close": round(mfe_pips, 4),
        "mfe_at_utc": mfe_at.isoformat() if mfe_at else None,
        "estimated_mfe_jpy_before_loss_close": estimated_mfe_jpy,
        "tp_distance_pips": round(tp_distance_pips, 4) if tp_distance_pips is not None else None,
        "tp_progress_before_loss_close": tp_progress,
        "profit_capture_missed_before_loss_close": capture_missed,
        "profit_capture_progress_threshold": TP_PROGRESS_CAPTURE_THRESHOLD,
        "profit_capture_counterfactual_exit": counterfactual.get("exit"),
        "profit_capture_counterfactual_pips": counterfactual.get("pips"),
        "profit_capture_counterfactual_jpy": counterfactual.get("jpy"),
        "profit_capture_counterfactual_net_improvement_jpy": counterfactual.get("net_improvement_jpy"),
        "profit_capture_counterfactual_pl_jpy": counterfactual.get("pl_jpy"),
        "repair_replay_triggered_before_loss_close": repair_replay.get("triggered"),
        "repair_replay_exit": repair_replay.get("exit"),
        "repair_replay_trigger_at_utc": repair_replay.get("trigger_at_utc"),
        "repair_replay_profit_pips": repair_replay.get("profit_pips"),
        "repair_replay_tp_progress": repair_replay.get("tp_progress"),
        "repair_replay_noise_floor_pips": repair_replay.get("noise_floor_pips"),
        "repair_replay_spread_pips": repair_replay.get("spread_pips"),
        "repair_replay_m1_atr_pips": repair_replay.get("m1_atr_pips"),
        "repair_replay_counterfactual_jpy": repair_replay.get("jpy"),
        "repair_replay_counterfactual_net_improvement_jpy": repair_replay.get("net_improvement_jpy"),
        "repair_replay_counterfactual_pl_jpy": repair_replay.get("pl_jpy"),
        "jpy_per_pip_estimate": round(jpy_per_pip, 6) if jpy_per_pip is not None else None,
        "candles_available": len(candles),
    }


def _profit_capture_counterfactual(
    close: _MarketClose,
    *,
    capture_missed: bool,
    tp_touch_at: datetime | None,
    tp_distance_pips: float | None,
    mfe_pips: float,
    jpy_per_pip: float | None,
) -> dict[str, Any]:
    if not capture_missed or jpy_per_pip is None:
        return {}
    if tp_touch_at is not None and tp_distance_pips is not None and tp_distance_pips > 0:
        exit_label = "TAKE_PROFIT_TOUCH"
        capture_pips = tp_distance_pips
    elif tp_distance_pips is not None and tp_distance_pips > 0:
        exit_label = "TP_PROGRESS_CAPTURE"
        capture_pips = min(mfe_pips, tp_distance_pips * TP_PROGRESS_CAPTURE_THRESHOLD)
    else:
        exit_label = "MFE_CAPTURE"
        capture_pips = mfe_pips
    if capture_pips <= 0:
        return {}
    capture_jpy = round(capture_pips * jpy_per_pip, 4)
    return {
        "exit": exit_label,
        "pips": round(capture_pips, 4),
        "jpy": capture_jpy,
        "pl_jpy": capture_jpy,
        "net_improvement_jpy": round(capture_jpy - close.realized_pl_jpy, 4),
    }


def _profit_capture_repair_replay(
    close: _MarketClose,
    candles: tuple[BidAskCandle, ...],
    *,
    tp_distance_pips: float | None,
    jpy_per_pip: float | None,
) -> dict[str, Any]:
    if not candles or tp_distance_pips is None or tp_distance_pips <= 0 or jpy_per_pip is None:
        return {"triggered": False}
    progress_gate = max(0.0, min(1.0, TP_PROGRESS_CAPTURE_THRESHOLD))
    for idx, candle in enumerate(candles):
        profit_pips = _favorable_delta(close.side, close.entry, candle) * _pip_factor(close.pair)
        if profit_pips <= 0:
            continue
        progress = profit_pips / tp_distance_pips
        if progress < progress_gate:
            continue
        spread_pips = _candle_spread_pips(close.pair, candle)
        atr_pips = _rolling_mid_range_pips(
            close.pair,
            candles,
            idx,
            period=TP_PROGRESS_REPAIR_ATR_PERIOD,
        )
        if spread_pips is None or spread_pips <= 0 or atr_pips is None or atr_pips <= 0:
            continue
        noise_floor = max(spread_pips * TP_PROGRESS_REPAIR_NOISE_MULT, atr_pips)
        if profit_pips < noise_floor:
            continue
        capture_jpy = round(profit_pips * jpy_per_pip, 4)
        return {
            "triggered": True,
            "exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
            "trigger_at_utc": candle.timestamp_utc.isoformat(),
            "profit_pips": round(profit_pips, 4),
            "tp_progress": round(progress, 4),
            "noise_floor_pips": round(noise_floor, 4),
            "spread_pips": round(spread_pips, 4),
            "m1_atr_pips": round(atr_pips, 4),
            "jpy": capture_jpy,
            "pl_jpy": capture_jpy,
            "net_improvement_jpy": round(capture_jpy - close.realized_pl_jpy, 4),
        }
    return {"triggered": False}


def _audit_market_close_counterfactual(
    close: _MarketClose,
    candles: tuple[BidAskCandle, ...],
    home_conversions: dict[str, float],
) -> dict[str, Any]:
    favorable_pips = 0.0
    favorable_at: datetime | None = None
    first_favorable_at: datetime | None = None
    adverse_pips = 0.0
    adverse_at: datetime | None = None
    first_adverse_at: datetime | None = None
    tp_touch_at: datetime | None = None
    sl_touch_at: datetime | None = None
    for candle in candles:
        favorable = _favorable_delta_from_price(close.side, close.close_price, candle) * _pip_factor(close.pair)
        adverse = _adverse_delta_from_price(close.side, close.close_price, candle) * _pip_factor(close.pair)
        if favorable > 0 and first_favorable_at is None:
            first_favorable_at = candle.timestamp_utc
        if adverse > 0 and first_adverse_at is None:
            first_adverse_at = candle.timestamp_utc
        if favorable > favorable_pips:
            favorable_pips = favorable
            favorable_at = candle.timestamp_utc
        if adverse > adverse_pips:
            adverse_pips = adverse
            adverse_at = candle.timestamp_utc
        if close.tp is not None and tp_touch_at is None and _tp_touched(close.side, close.tp, candle):
            tp_touch_at = candle.timestamp_utc
        if close.sl is not None and sl_touch_at is None and _sl_touched(close.side, close.sl, candle):
            sl_touch_at = candle.timestamp_utc
    jpy_per_pip = _jpy_per_pip(
        close.pair,
        close.units,
        home_conversions=home_conversions,
        raw=close.close_raw,
    )
    favorable_jpy = round(favorable_pips * jpy_per_pip, 4) if jpy_per_pip is not None else None
    adverse_jpy = round(adverse_pips * jpy_per_pip, 4) if jpy_per_pip is not None else None
    path_label = _market_close_path_label(
        realized_pl_jpy=close.realized_pl_jpy,
        favorable_pips=favorable_pips,
        adverse_pips=adverse_pips,
        candles_available=len(candles),
    )
    return {
        "trade_id": close.trade_id,
        "lane_id": close.lane_id,
        "pair": close.pair,
        "side": close.side,
        "units": close.units,
        "entry": close.entry,
        "close_price": close.close_price,
        "tp": close.tp,
        "sl": close.sl,
        "exit_reason": close.exit_reason,
        "gateway_action": close.gateway_action,
        "realized_pl_jpy": round(close.realized_pl_jpy, 4),
        "fill_at_utc": close.fill_at_utc.isoformat(),
        "close_at_utc": close.close_at_utc.isoformat(),
        "post_close_path_label": path_label,
        "post_close_favorable_pips": round(favorable_pips, 4),
        "post_close_favorable_at_utc": favorable_at.isoformat() if favorable_at else None,
        "first_post_close_favorable_minutes": _minutes_between(close.close_at_utc, first_favorable_at),
        "estimated_post_close_favorable_jpy": favorable_jpy,
        "post_close_adverse_pips": round(adverse_pips, 4),
        "post_close_adverse_at_utc": adverse_at.isoformat() if adverse_at else None,
        "first_post_close_adverse_minutes": _minutes_between(close.close_at_utc, first_adverse_at),
        "estimated_post_close_adverse_jpy": adverse_jpy,
        "tp_touched_after_market_close": tp_touch_at is not None,
        "tp_touch_minutes_after_market_close": _minutes_between(close.close_at_utc, tp_touch_at),
        "sl_touched_after_market_close": sl_touch_at is not None,
        "sl_touch_minutes_after_market_close": _minutes_between(close.close_at_utc, sl_touch_at),
        "jpy_per_pip_estimate": round(jpy_per_pip, 6) if jpy_per_pip is not None else None,
        "candles_available": len(candles),
    }


def _summary(
    canceled_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    market_close_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    canceled_entry = [row for row in canceled_rows if row.get("entry_touched_after_cancel")]
    canceled_positive = [row for row in canceled_rows if float(row.get("mfe_pips_after_cancel_entry") or 0.0) > 0]
    canceled_tp = [row for row in canceled_rows if row.get("tp_touched_after_cancel")]
    loss_positive = [row for row in loss_rows if row.get("had_positive_mfe_before_loss_close")]
    loss_tp = [row for row in loss_rows if row.get("tp_touched_before_loss_close")]
    loss_capture_missed = [
        row for row in loss_rows if row.get("profit_capture_missed_before_loss_close")
    ]
    stop_loss_capture_missed = [
        row for row in loss_capture_missed if str(row.get("exit_reason") or "").upper() == "STOP_LOSS_ORDER"
    ]
    lag_values = [
        float(row["decision_lag_minutes_after_first_positive"])
        for row in loss_positive
        if row.get("decision_lag_minutes_after_first_positive") is not None
    ]
    market_continued = [
        row
        for row in market_close_rows
        if row.get("post_close_path_label")
        in {"PROFIT_CLOSE_LEFT_RUNNER_UPSIDE", "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE"}
    ]
    market_adverse = [
        row
        for row in market_close_rows
        if row.get("post_close_path_label")
        in {"PROFIT_CLOSE_AVOIDED_GIVEBACK", "LOSS_CLOSE_CONTAINED_RISK"}
    ]
    profit_market = [row for row in market_close_rows if float(row.get("realized_pl_jpy") or 0.0) >= 0.0]
    loss_market = [row for row in market_close_rows if float(row.get("realized_pl_jpy") or 0.0) < 0.0]
    profit_left_runner = [
        row for row in profit_market if row.get("post_close_path_label") == "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE"
    ]
    profit_avoided = [
        row for row in profit_market if row.get("post_close_path_label") == "PROFIT_CLOSE_AVOIDED_GIVEBACK"
    ]
    loss_premature = [
        row for row in loss_market if row.get("post_close_path_label") == "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE"
    ]
    loss_contained = [
        row for row in loss_market if row.get("post_close_path_label") == "LOSS_CLOSE_CONTAINED_RISK"
    ]
    market_tp_after = [row for row in market_close_rows if row.get("tp_touched_after_market_close")]
    market_sl_after = [row for row in market_close_rows if row.get("sl_touched_after_market_close")]
    actual_loss_close_pl = _sum_known(loss_rows, "realized_pl_jpy")
    counterfactual_loss_close_pl = _counterfactual_loss_close_pl(loss_rows)
    repair_replay_rows = [
        row for row in loss_rows if row.get("repair_replay_triggered_before_loss_close")
    ]
    repair_replay_pl = _repair_replay_loss_close_pl(loss_rows)
    return {
        "canceled_orders_audited": len(canceled_rows),
        "canceled_entry_touched_after_cancel": len(canceled_entry),
        "canceled_entry_touched_after_cancel_rate": _rate(len(canceled_entry), len(canceled_rows)),
        "canceled_positive_after_cancel_entry": len(canceled_positive),
        "canceled_positive_after_cancel_entry_rate": _rate(len(canceled_positive), len(canceled_rows)),
        "canceled_tp_touched_after_cancel": len(canceled_tp),
        "canceled_tp_touched_after_cancel_rate": _rate(len(canceled_tp), len(canceled_rows)),
        "canceled_estimated_missed_mfe_jpy": _sum_known(canceled_rows, "estimated_missed_mfe_jpy"),
        "loss_closes_audited": len(loss_rows),
        "loss_closes_had_positive_mfe": len(loss_positive),
        "loss_closes_had_positive_mfe_rate": _rate(len(loss_positive), len(loss_rows)),
        "loss_closes_tp_touched_before_close": len(loss_tp),
        "loss_closes_tp_touched_before_close_rate": _rate(len(loss_tp), len(loss_rows)),
        "loss_close_estimated_mfe_jpy": _sum_known(loss_rows, "estimated_mfe_jpy_before_loss_close"),
        "loss_closes_profit_capture_missed": len(loss_capture_missed),
        "loss_closes_profit_capture_missed_rate": _rate(len(loss_capture_missed), len(loss_rows)),
        "stop_loss_closes_profit_capture_missed": len(stop_loss_capture_missed),
        "loss_close_estimated_capture_gap_jpy": _sum_known(
            loss_capture_missed,
            "estimated_mfe_jpy_before_loss_close",
        ),
        "loss_close_actual_pl_jpy": actual_loss_close_pl,
        "loss_close_counterfactual_profit_capture_pl_jpy": counterfactual_loss_close_pl,
        "loss_close_counterfactual_profit_capture_delta_jpy": (
            round(counterfactual_loss_close_pl - actual_loss_close_pl, 4)
            if counterfactual_loss_close_pl is not None and actual_loss_close_pl is not None
            else None
        ),
        "loss_close_counterfactual_profit_capture_jpy": _sum_known(
            loss_capture_missed,
            "profit_capture_counterfactual_jpy",
        ),
        "loss_closes_repair_replay_triggered": len(repair_replay_rows),
        "loss_closes_repair_replay_triggered_rate": _rate(len(repair_replay_rows), len(loss_rows)),
        "loss_close_repair_replay_profit_capture_jpy": _sum_known(
            repair_replay_rows,
            "repair_replay_counterfactual_jpy",
        ),
        "loss_close_repair_replay_actual_pl_jpy": actual_loss_close_pl,
        "loss_close_repair_replay_counterfactual_pl_jpy": repair_replay_pl,
        "loss_close_repair_replay_delta_jpy": (
            round(repair_replay_pl - actual_loss_close_pl, 4)
            if repair_replay_pl is not None and actual_loss_close_pl is not None
            else None
        ),
        "avg_decision_lag_minutes_after_first_positive": round(sum(lag_values) / len(lag_values), 2) if lag_values else None,
        "max_decision_lag_minutes_after_first_positive": round(max(lag_values), 2) if lag_values else None,
        "market_closes_audited": len(market_close_rows),
        "market_closes_post_close_continued": len(market_continued),
        "market_closes_post_close_continued_rate": _rate(len(market_continued), len(market_close_rows)),
        "market_closes_post_close_adverse": len(market_adverse),
        "market_closes_post_close_adverse_rate": _rate(len(market_adverse), len(market_close_rows)),
        "market_closes_tp_touched_after_close": len(market_tp_after),
        "market_closes_tp_touched_after_close_rate": _rate(len(market_tp_after), len(market_close_rows)),
        "market_closes_sl_touched_after_close": len(market_sl_after),
        "market_closes_sl_touched_after_close_rate": _rate(len(market_sl_after), len(market_close_rows)),
        "market_close_estimated_followthrough_jpy": _sum_known(
            market_continued,
            "estimated_post_close_favorable_jpy",
        ),
        "market_close_estimated_avoided_adverse_jpy": _sum_known(
            market_adverse,
            "estimated_post_close_adverse_jpy",
        ),
        "profit_market_closes_audited": len(profit_market),
        "profit_market_closes_left_runner_upside": len(profit_left_runner),
        "profit_market_closes_avoided_giveback": len(profit_avoided),
        "loss_market_closes_audited": len(loss_market),
        "loss_market_closes_may_have_been_premature": len(loss_premature),
        "loss_market_closes_contained_risk": len(loss_contained),
    }


def _counterfactual_loss_close_pl(loss_rows: list[dict[str, Any]]) -> float | None:
    if not loss_rows:
        return None
    total = 0.0
    seen = False
    for row in loss_rows:
        value = row.get("profit_capture_counterfactual_pl_jpy")
        if value is None:
            value = row.get("realized_pl_jpy")
        if value is None:
            continue
        total += float(value)
        seen = True
    return round(total, 4) if seen else None


def _repair_replay_loss_close_pl(loss_rows: list[dict[str, Any]]) -> float | None:
    if not loss_rows:
        return None
    total = 0.0
    seen = False
    for row in loss_rows:
        value = row.get("repair_replay_counterfactual_pl_jpy")
        if value is None:
            value = row.get("realized_pl_jpy")
        if value is None:
            continue
        total += float(value)
        seen = True
    return round(total, 4) if seen else None


def _canceled_order_regret_by_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        pair = str(row.get("pair") or "UNKNOWN")
        side = str(row.get("side") or "UNKNOWN")
        method = _lane_method(str(row.get("lane_id") or ""))
        order_type = str(row.get("order_type") or "UNKNOWN")
        groups.setdefault((pair, side, method, order_type), []).append(row)

    items: list[dict[str, Any]] = []
    for (pair, side, method, order_type), bucket in groups.items():
        orders = len(bucket)
        entry_touched = [row for row in bucket if row.get("entry_touched_after_cancel")]
        positive = [
            row
            for row in bucket
            if float(row.get("mfe_pips_after_cancel_entry") or 0.0) > 0.0
        ]
        tp_touched = [row for row in bucket if row.get("tp_touched_after_cancel")]
        missed_mfe_jpy = _sum_known(bucket, "estimated_missed_mfe_jpy")
        avg_entry_touch = _avg_known(entry_touched, "entry_touch_after_cancel_minutes")
        avg_tp_touch = _avg_known(tp_touched, "tp_touch_after_cancel_minutes")
        if tp_touched:
            priority_class = "PRESERVE_PENDING_THESIS_TP_TOUCHED"
            rank = 0
            next_action = (
                "review cancel rule/TTL before canceling this pending shape; canceled orders later "
                "reached broker TP in the audit window"
            )
        elif entry_touched and positive:
            priority_class = "REPRICE_OR_EXTEND_TTL_ENTRY_TOUCHED"
            rank = 1
            next_action = (
                "repair pending TTL/reprice logic; canceled orders later touched entry and produced "
                "positive MFE"
            )
        elif entry_touched:
            priority_class = "ENTRY_TOUCHED_NO_POSITIVE_MFE"
            rank = 2
            next_action = "inspect fill economics before preserving this pending shape"
        else:
            priority_class = "LOW_CANCEL_REGRET"
            rank = 3
            next_action = "current cancel behavior has no post-cancel entry touch evidence in this window"
        items.append(
            {
                "evidence_ref": f"timing:canceled_shape:{pair}:{side}:{method}:{order_type}",
                "pair": pair,
                "side": side,
                "method": method,
                "order_type": order_type,
                "priority_class": priority_class,
                "next_action": next_action,
                "orders": orders,
                "entry_touched_after_cancel": len(entry_touched),
                "entry_touch_after_cancel_rate": _rate(len(entry_touched), orders),
                "positive_after_cancel_entry": len(positive),
                "positive_after_cancel_entry_rate": _rate(len(positive), orders),
                "tp_touched_after_cancel": len(tp_touched),
                "tp_touched_after_cancel_rate": _rate(len(tp_touched), orders),
                "estimated_missed_mfe_jpy": missed_mfe_jpy,
                "avg_entry_touch_after_cancel_minutes": avg_entry_touch,
                "avg_tp_touch_after_cancel_minutes": avg_tp_touch,
                "_sort_rank": rank,
            }
        )

    sorted_items = sorted(
        items,
        key=lambda item: (
            int(item.get("_sort_rank") or 0),
            -float(item.get("estimated_missed_mfe_jpy") or 0.0),
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
            str(item.get("order_type") or ""),
        ),
    )
    for item in sorted_items:
        item.pop("_sort_rank", None)
    return {
        "basis": "canceled pending orders grouped by pair|side|method|order_type",
        "total_shapes": len(sorted_items),
        "items": sorted_items[:TIMING_SHAPE_ROLLUP_LIMIT],
    }


def _avg_known(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 2) if values else None


def _lane_method(lane_id: str) -> str:
    parts = [part for part in str(lane_id or "").split(":") if part]
    if len(parts) >= 4:
        return parts[3]
    return "UNKNOWN"


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    summary = payload.get("summary") or {}
    lines = [
        "# Execution Timing Audit",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Window UTC: `{(payload.get('window') or {}).get('from_utc')}` to `{(payload.get('window') or {}).get('to_utc')}`",
        f"- Precision: `{(payload.get('precision') or {}).get('price_basis')}`",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "canceled_orders_audited",
        "canceled_entry_touched_after_cancel",
        "canceled_entry_touched_after_cancel_rate",
        "canceled_positive_after_cancel_entry",
        "canceled_positive_after_cancel_entry_rate",
        "canceled_tp_touched_after_cancel",
        "canceled_tp_touched_after_cancel_rate",
        "canceled_estimated_missed_mfe_jpy",
        "loss_closes_audited",
        "loss_closes_had_positive_mfe",
        "loss_closes_had_positive_mfe_rate",
        "loss_closes_tp_touched_before_close",
        "loss_closes_tp_touched_before_close_rate",
        "loss_close_estimated_mfe_jpy",
        "loss_closes_profit_capture_missed",
        "loss_closes_profit_capture_missed_rate",
        "stop_loss_closes_profit_capture_missed",
        "loss_close_estimated_capture_gap_jpy",
        "loss_close_actual_pl_jpy",
        "loss_close_counterfactual_profit_capture_pl_jpy",
        "loss_close_counterfactual_profit_capture_delta_jpy",
        "loss_close_counterfactual_profit_capture_jpy",
        "loss_closes_repair_replay_triggered",
        "loss_closes_repair_replay_triggered_rate",
        "loss_close_repair_replay_profit_capture_jpy",
        "loss_close_repair_replay_actual_pl_jpy",
        "loss_close_repair_replay_counterfactual_pl_jpy",
        "loss_close_repair_replay_delta_jpy",
        "avg_decision_lag_minutes_after_first_positive",
        "max_decision_lag_minutes_after_first_positive",
        "market_closes_audited",
        "market_closes_post_close_continued",
        "market_closes_post_close_continued_rate",
        "market_closes_post_close_adverse",
        "market_closes_post_close_adverse_rate",
        "market_closes_tp_touched_after_close",
        "market_closes_tp_touched_after_close_rate",
        "market_closes_sl_touched_after_close",
        "market_closes_sl_touched_after_close_rate",
        "market_close_estimated_followthrough_jpy",
        "market_close_estimated_avoided_adverse_jpy",
        "profit_market_closes_audited",
        "profit_market_closes_left_runner_upside",
        "profit_market_closes_avoided_giveback",
        "loss_market_closes_audited",
        "loss_market_closes_may_have_been_premature",
        "loss_market_closes_contained_risk",
    ):
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    shape_rollup = payload.get("canceled_order_regret_by_shape")
    shape_items = (
        shape_rollup.get("items")
        if isinstance(shape_rollup, dict)
        else []
    )
    if shape_items:
        lines.extend(
            [
                "",
                "## Canceled Order Regret By Shape",
                "",
                "| pair | side | method | type | priority | orders | entry touch | TP touch | missed MFE JPY |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in shape_items:
            lines.append(
                "| `{pair}` | `{side}` | `{method}` | `{order_type}` | `{priority}` | {orders} | {entry} | {tp} | {mfe} |".format(
                    pair=row.get("pair"),
                    side=row.get("side"),
                    method=row.get("method"),
                    order_type=row.get("order_type"),
                    priority=row.get("priority_class"),
                    orders=row.get("orders"),
                    entry=row.get("entry_touched_after_cancel"),
                    tp=row.get("tp_touched_after_cancel"),
                    mfe=row.get("estimated_missed_mfe_jpy"),
                )
            )
    lines.extend(["", "## Top Canceled Order Regrets", "", "| Order | Lane | Pair | Side | Entry touch min | TP touch min | MFE pips | Est MFE JPY |", "|---|---|---|---|---:|---:|---:|---:|"])
    top_canceled = sorted(
        payload.get("canceled_order_regrets") or [],
        key=lambda row: float(row.get("estimated_missed_mfe_jpy") or 0.0),
        reverse=True,
    )[:20]
    for row in top_canceled:
        lines.append(
            "| `{order_id}` | `{lane_id}` | `{pair}` | `{side}` | `{entry_min}` | `{tp_min}` | `{mfe}` | `{jpy}` |".format(
                order_id=row.get("order_id"),
                lane_id=row.get("lane_id") or "",
                pair=row.get("pair"),
                side=row.get("side"),
                entry_min=row.get("entry_touch_after_cancel_minutes"),
                tp_min=row.get("tp_touch_after_cancel_minutes"),
                mfe=row.get("mfe_pips_after_cancel_entry"),
                jpy=row.get("estimated_missed_mfe_jpy"),
            )
        )
    lines.extend(["", "## Top Loss Close Timing Regrets", "", "| Trade | Lane | Pair | Side | Exit | PL JPY | First plus min | Lag min | MFE pips | TP progress | Capture missed | Repair replay | Repair JPY | Repair delta | Est MFE JPY | TP touched |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|"])
    top_loss = sorted(
        payload.get("loss_close_regrets") or [],
        key=lambda row: float(row.get("estimated_mfe_jpy_before_loss_close") or 0.0),
        reverse=True,
    )[:20]
    for row in top_loss:
        lines.append(
            "| `{trade}` | `{lane}` | `{pair}` | `{side}` | `{exit}` | `{pl}` | `{first}` | `{lag}` | `{mfe}` | `{progress}` | `{missed}` | `{repair}` | `{repair_jpy}` | `{repair_delta}` | `{jpy}` | `{tp}` |".format(
                trade=row.get("trade_id"),
                lane=row.get("lane_id") or "",
                pair=row.get("pair"),
                side=row.get("side"),
                exit=row.get("exit_reason"),
                pl=row.get("realized_pl_jpy"),
                first=row.get("first_positive_minutes_after_fill"),
                lag=row.get("decision_lag_minutes_after_first_positive"),
                mfe=row.get("mfe_pips_before_loss_close"),
                progress=row.get("tp_progress_before_loss_close"),
                missed=row.get("profit_capture_missed_before_loss_close"),
                repair=row.get("repair_replay_triggered_before_loss_close"),
                repair_jpy=row.get("repair_replay_counterfactual_jpy"),
                repair_delta=row.get("repair_replay_counterfactual_net_improvement_jpy"),
                jpy=row.get("estimated_mfe_jpy_before_loss_close"),
                tp=row.get("tp_touched_before_loss_close"),
            )
        )
    lines.extend(
        [
            "",
            "## Top Market Close Counterfactuals",
            "",
            "| Trade | Lane | Pair | Side | Gateway | PL JPY | Label | Fav pips | Fav JPY | Adv pips | Adv JPY | TP after | SL after |",
            "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    top_market = sorted(
        payload.get("market_close_counterfactuals") or [],
        key=lambda row: max(
            float(row.get("estimated_post_close_favorable_jpy") or 0.0),
            float(row.get("estimated_post_close_adverse_jpy") or 0.0),
        ),
        reverse=True,
    )[:20]
    for row in top_market:
        lines.append(
            "| `{trade}` | `{lane}` | `{pair}` | `{side}` | `{gateway}` | `{pl}` | `{label}` | `{fav}` | `{fav_jpy}` | `{adv}` | `{adv_jpy}` | `{tp}` | `{sl}` |".format(
                trade=row.get("trade_id"),
                lane=row.get("lane_id") or "",
                pair=row.get("pair"),
                side=row.get("side"),
                gateway=row.get("gateway_action") or "",
                pl=row.get("realized_pl_jpy"),
                label=row.get("post_close_path_label"),
                fav=row.get("post_close_favorable_pips"),
                fav_jpy=row.get("estimated_post_close_favorable_jpy"),
                adv=row.get("post_close_adverse_pips"),
                adv_jpy=row.get("estimated_post_close_adverse_jpy"),
                tp=row.get("tp_touched_after_market_close"),
                sl=row.get("sl_touched_after_market_close"),
            )
        )
    errors = payload.get("fetch_errors") or []
    if errors:
        lines.extend(["", "## Fetch Errors", ""])
        for item in errors[:40]:
            lines.append(f"- `{item.get('pair')}` `{item.get('from_utc')}`..`{item.get('to_utc')}`: {item.get('error')}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def _protections_by_trade(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trade_id = str(row.get("trade_id") or "")
        if not trade_id:
            continue
        raw = _json_obj(row.get("raw_json"))
        order_type = str(raw.get("type") or "").upper()
        if order_type not in {"TAKE_PROFIT_ORDER", "STOP_LOSS_ORDER"}:
            continue
        out.setdefault(trade_id, []).append(row)
    return {key: tuple(sorted(value, key=lambda row: _parse_utc(row.get("ts_utc")) or datetime.min.replace(tzinfo=timezone.utc))) for key, value in out.items()}


def _latest_protection_before(rows: tuple[dict[str, Any], ...], close_at: datetime) -> tuple[float | None, float | None]:
    tp: float | None = None
    sl: float | None = None
    for row in rows:
        ts = _parse_utc(row.get("ts_utc"))
        if ts is None or ts > close_at:
            continue
        raw = _json_obj(row.get("raw_json"))
        order_type = str(raw.get("type") or "").upper()
        price = _float(row.get("price")) or _float(raw.get("price"))
        if order_type == "TAKE_PROFIT_ORDER":
            tp = price
        elif order_type == "STOP_LOSS_ORDER":
            sl = price
    return tp, sl


def _tp_distance_pips(close: _MarketClose) -> float | None:
    if close.tp is None:
        return None
    return abs(close.tp - close.entry) * _pip_factor(close.pair)


def _gateway_closes_by_trade(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trade_id = str(row.get("trade_id") or "")
        if not trade_id:
            continue
        out.setdefault(trade_id, []).append(row)
    return {key: tuple(sorted(value, key=lambda row: _parse_utc(row.get("ts_utc")) or datetime.min.replace(tzinfo=timezone.utc))) for key, value in out.items()}


def _latest_gateway_close_action_before(rows: tuple[dict[str, Any], ...], close_at: datetime) -> str | None:
    action: str | None = None
    for row in rows:
        ts = _parse_utc(row.get("ts_utc"))
        if ts is None or ts > close_at:
            continue
        text = str(row.get("exit_reason") or "").strip()
        if text:
            action = text
    return action


def _entry_touched(order_type: str, side: str, entry: float, candle: BidAskCandle) -> bool:
    if order_type == "LIMIT_ORDER":
        return candle.ask_low <= entry if side == "LONG" else candle.bid_high >= entry
    if order_type == "STOP_ORDER":
        return candle.ask_high >= entry if side == "LONG" else candle.bid_low <= entry
    return False


def _favorable_delta(side: str, entry: float, candle: BidAskCandle) -> float:
    return _favorable_delta_from_price(side, entry, candle)


def _favorable_delta_from_price(side: str, price: float, candle: BidAskCandle) -> float:
    if side == "LONG":
        return max(0.0, candle.bid_high - price)
    return max(0.0, price - candle.ask_low)


def _adverse_delta_from_price(side: str, price: float, candle: BidAskCandle) -> float:
    if side == "LONG":
        return max(0.0, price - candle.bid_low)
    return max(0.0, candle.ask_high - price)


def _market_close_path_label(
    *,
    realized_pl_jpy: float,
    favorable_pips: float,
    adverse_pips: float,
    candles_available: int,
) -> str:
    if candles_available <= 0:
        return "UNMEASURED"
    if favorable_pips > adverse_pips:
        return "PROFIT_CLOSE_LEFT_RUNNER_UPSIDE" if realized_pl_jpy >= 0.0 else "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE"
    if adverse_pips > favorable_pips:
        return "PROFIT_CLOSE_AVOIDED_GIVEBACK" if realized_pl_jpy >= 0.0 else "LOSS_CLOSE_CONTAINED_RISK"
    return "MIXED_OR_FLAT_AFTER_CLOSE"


def _tp_touched(side: str, tp: float, candle: BidAskCandle) -> bool:
    return candle.bid_high >= tp if side == "LONG" else candle.ask_low <= tp


def _sl_touched(side: str, sl: float, candle: BidAskCandle) -> bool:
    return candle.bid_low <= sl if side == "LONG" else candle.ask_high >= sl


def _candle_spread_pips(pair: str, candle: BidAskCandle) -> float | None:
    factor = _pip_factor(pair)
    spreads = [
        (candle.ask_high - candle.bid_high) * factor,
        (candle.ask_low - candle.bid_low) * factor,
    ]
    positive = [value for value in spreads if value > 0]
    return max(positive) if positive else None


def _rolling_mid_range_pips(
    pair: str,
    candles: tuple[BidAskCandle, ...],
    idx: int,
    *,
    period: int,
) -> float | None:
    factor = _pip_factor(pair)
    start = max(0, idx - max(1, period) + 1)
    ranges: list[float] = []
    for candle in candles[start : idx + 1]:
        mid_high = (candle.bid_high + candle.ask_high) / 2.0
        mid_low = (candle.bid_low + candle.ask_low) / 2.0
        if mid_high > mid_low:
            ranges.append((mid_high - mid_low) * factor)
    if not ranges:
        return None
    return sum(ranges) / len(ranges)


def _jpy_per_pip(
    pair: str,
    units: int,
    *,
    home_conversions: dict[str, float],
    raw: dict[str, Any] | None,
) -> float | None:
    if units <= 0:
        return None
    if pair.endswith("_JPY"):
        return units / 100.0
    quote_ccy = pair.split("_", 1)[1] if "_" in pair else ""
    factor = home_conversions.get(quote_ccy)
    if factor is None and raw:
        factor = _quote_home_factor_from_raw(raw)
    if factor is None or factor <= 0:
        return None
    return (units / _pip_factor(pair)) * factor


def _quote_home_factor_from_raw(raw: dict[str, Any]) -> float | None:
    for key in ("gainQuoteHomeConversionFactor", "lossQuoteHomeConversionFactor"):
        value = _float(raw.get(key))
        if value is not None and value > 0:
            return value
    factors = raw.get("homeConversionFactors") if isinstance(raw.get("homeConversionFactors"), dict) else {}
    for key in ("gainQuoteHome", "lossQuoteHome"):
        item = factors.get(key) if isinstance(factors.get(key), dict) else {}
        value = _float(item.get("factor"))
        if value is not None and value > 0:
            return value
    return None


def _home_conversions_from_snapshot(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    raw = payload.get("home_conversions")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        number = _float(value)
        if number is not None and number > 0:
            out[str(key).upper()] = number
    return out


def _merge_intervals(intervals: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _chunk_interval(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = start
    step = timedelta(hours=DEFAULT_CANDLE_CHUNK_HOURS)
    while current < end:
        nxt = min(end, current + step)
        yield current, nxt
        current = nxt


def _candles_between(candles: tuple[BidAskCandle, ...], start: datetime, end: datetime) -> tuple[BidAskCandle, ...]:
    if end < start:
        return ()
    return tuple(c for c in candles if start <= c.timestamp_utc <= end)


def _candle_key(candle: BidAskCandle) -> tuple[datetime, float, float, float, float]:
    return (candle.timestamp_utc, candle.bid_high, candle.bid_low, candle.ask_high, candle.ask_low)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _json_obj(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _nested_price(value: object) -> float | None:
    return _float(value.get("price")) if isinstance(value, dict) else None


def _side_from_units(value: object) -> str | None:
    number = _float(value)
    if number is None or number == 0:
        return None
    return "LONG" if number > 0 else "SHORT"


def _float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _sum_known(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values), 4) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _minutes_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 60.0, 2)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _granularity_delta(granularity: str) -> timedelta:
    text = granularity.upper().strip()
    if text.startswith("S"):
        return timedelta(seconds=max(1, int(text[1:] or "5")))
    if text.startswith("M"):
        return timedelta(minutes=max(1, int(text[1:] or "1")))
    if text.startswith("H"):
        return timedelta(hours=max(1, int(text[1:] or "1")))
    if text.startswith("D"):
        return timedelta(days=1)
    return timedelta(minutes=1)


def _first_complete_candle_start_at_or_after(value: datetime, granularity_delta: timedelta) -> datetime:
    value = _normalize_utc(value)
    seconds = granularity_delta.total_seconds()
    if seconds <= 0:
        return value
    epoch_seconds = value.timestamp()
    remainder = epoch_seconds % seconds
    if remainder == 0:
        return value
    return value + timedelta(seconds=seconds - remainder)


def _last_complete_candle_start_before(value: datetime, granularity_delta: timedelta) -> datetime:
    value = _normalize_utc(value)
    seconds = granularity_delta.total_seconds()
    if seconds <= 0:
        return value
    epoch_seconds = value.timestamp()
    candle_start = epoch_seconds - (epoch_seconds % seconds)
    if candle_start + seconds <= epoch_seconds:
        return datetime.fromtimestamp(candle_start, tz=timezone.utc)
    return datetime.fromtimestamp(candle_start - seconds, tz=timezone.utc)


def _parse_utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, rest = text.split(".", 1)
        if "+" in rest:
            frac, zone = rest.split("+", 1)
            text = f"{head}.{frac[:6]}+{zone}"
        elif "-" in rest:
            frac, zone = rest.split("-", 1)
            text = f"{head}.{frac[:6]}-{zone}"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _format_oanda_time(value: datetime) -> str:
    return _normalize_utc(value).isoformat().replace("+00:00", "Z")
