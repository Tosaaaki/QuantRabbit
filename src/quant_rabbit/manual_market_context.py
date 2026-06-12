"""Audit technical context around the operator's 2025 manual trades.

The manual-history precedent proves that a 200%+ funding-adjusted 30-day
window happened. This module asks the next question: what technical state was
the operator actually trading into? The output is advisory evidence for using
the precedent; it does not grant live permission or bypass current gates.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_rabbit.analysis.candles import Candle, fetch_candles_between
from quant_rabbit.analysis.indicators import compute_indicators
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.paths import (
    DEFAULT_MANUAL_HISTORY_2025,
    DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT_REPORT,
)


# Indicator windows are counts of historical bars supplied to the same
# QuantRabbit indicator stack used by current pair-charts. They are evidence
# windows, not trading thresholds; missing history emits warnings/blocks.
LOOKBACK_BARS_BY_TF: dict[str, int] = {"M5": 220, "H1": 220}

# Keep requests below OANDA's documented 5000-candle ceiling. The buffer avoids
# edge failures when includeFirst duplicates a boundary candle. This is a
# transport pagination constant, not a market parameter.
OANDA_CANDLE_CHUNK_LIMIT = 4500

# Descriptive buckets used to summarize the operator's observed hold-time
# profile. The 12h boundary mirrors the previously mined margin-closeout cliff;
# these buckets do not authorize live exits by themselves.
HOLD_BUCKET_HOURS = (0.5, 2.0, 12.0)


@dataclass(frozen=True)
class ManualMarketContextSummary:
    output_path: Path
    report_path: Path
    status: str
    analyzed_trades: int
    blockers: int
    warnings: int
    best_h1_alignment: str | None
    worst_h1_alignment: str | None


def build_manual_market_context_audit(
    *,
    manual_history_path: Path = DEFAULT_MANUAL_HISTORY_2025,
    output_path: Path = DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    report_path: Path = DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT_REPORT,
    pair: str = "USD_JPY",
    candles_by_tf: Mapping[str, Iterable[Candle]] | None = None,
    client: OandaReadOnlyClient | None = None,
    now: datetime | None = None,
    max_trades: int | None = None,
) -> ManualMarketContextSummary:
    clock = now or datetime.now(timezone.utc)
    checks: list[dict[str, Any]] = []
    manual_payload, manual_error = _read_json(manual_history_path)
    checks.append(
        _check(
            "manual_history_readable",
            "PASS" if manual_payload is not None else "BLOCK",
            f"manual history artifact readable: {manual_history_path}"
            if manual_payload is not None
            else f"manual history artifact missing/unreadable: {manual_error or manual_history_path}",
            {"path": str(manual_history_path), "error": manual_error},
        )
    )

    trades = _manual_trades(manual_payload or {}, pair=pair)
    if max_trades is not None:
        trades = trades[: max(0, int(max_trades))]
    checks.append(
        _check(
            "manual_trades_for_pair",
            "PASS" if trades else "BLOCK",
            f"manual history exposes {len(trades)} {pair} trade exits"
            if trades
            else f"manual history exposes no {pair} trade exits",
            {"pair": pair, "trades": len(trades)},
        )
    )

    candles, candle_checks = _resolve_candles(
        trades=trades,
        pair=pair,
        candles_by_tf=candles_by_tf,
        client=client,
    )
    checks.extend(candle_checks)

    contexts = [_trade_context(trade, candles) for trade in trades]
    analyzed = [row for row in contexts if row is not None]
    missing_context = len(trades) - len(analyzed)
    checks.append(
        _check(
            "technical_context_coverage",
            "PASS" if analyzed else "BLOCK",
            f"technical context computed for {len(analyzed)}/{len(trades)} trade exits"
            if analyzed
            else "technical context could not be computed for any manual trade",
            {
                "trades": len(trades),
                "analyzed_trades": len(analyzed),
                "missing_context": missing_context,
            },
        )
    )
    if missing_context and analyzed:
        checks.append(
            _check(
                "technical_context_missing_rows",
                "WARN",
                f"{missing_context} manual trade exits lacked enough pre-entry candles",
                {"missing_context": missing_context},
                severity="WARN",
            )
        )

    profile = _technical_profile(analyzed)
    bounded_rows = [row for row in analyzed if not _is_unbounded_tail(row)]
    bounded_profile = _technical_profile(bounded_rows)
    excluded_tail_profile = _technical_profile([row for row in analyzed if _is_unbounded_tail(row)])
    position_building_profile = _position_building_profile(trades)
    best_h1 = _best_bucket(bounded_profile["by_h1_alignment"])
    worst_h1 = _worst_bucket(bounded_profile["by_h1_alignment"])
    if best_h1 and worst_h1:
        checks.append(
            _check(
                "bounded_h1_alignment_edge_extracted",
                "PASS",
                "bounded manual trades are split by entry-side alignment with H1 trend",
                {"best": best_h1, "worst": worst_h1},
            )
        )
    raw_best_h1 = _best_bucket(profile["by_h1_alignment"])
    if raw_best_h1 and best_h1 and raw_best_h1.get("bucket") != best_h1.get("bucket"):
        checks.append(
            _check(
                "raw_tail_differs_from_bounded_replay",
                "WARN",
                "raw H1 alignment net is dominated by unbounded long-hold/margin-tail rows; use bounded replay profile for precedent aggression",
                {"raw_best": raw_best_h1, "bounded_best": best_h1},
                severity="WARN",
            )
        )
    multi_entry_clusters = int(
        (position_building_profile.get("overall") or {}).get("multi_entry_clusters") or 0
    )
    adverse_add_clusters = int(
        (position_building_profile.get("adverse_adds") or {}).get("clusters") or 0
    )
    checks.append(
        _check(
            "manual_position_building_reconstructed",
            "PASS",
            (
                f"manual history reconstructs {multi_entry_clusters} overlapping same-side position-building cluster(s)"
                if multi_entry_clusters
                else "manual history has no overlapping same-side position-building clusters"
            ),
            {
                "multi_entry_clusters": multi_entry_clusters,
                "adverse_add_clusters": adverse_add_clusters,
            },
            severity="INFO",
        )
    )
    if adverse_add_clusters:
        adverse_net = _maybe_float((position_building_profile.get("adverse_adds") or {}).get("net_jpy"))
        checks.append(
            _check(
                "manual_nanpin_outcome",
                "PASS" if adverse_net is not None and adverse_net > 0 else "WARN",
                (
                    f"manual averaging-into-adverse clusters net {adverse_net:.1f} JPY"
                    if adverse_net is not None
                    else "manual averaging-into-adverse clusters detected but net outcome is unknown"
                ),
                position_building_profile.get("adverse_adds") or {},
                severity="INFO" if adverse_net is not None and adverse_net > 0 else "WARN",
            )
        )

    blockers = [item for item in checks if item["severity"] == "BLOCK" or item["status"] == "BLOCK"]
    warnings = [item for item in checks if item["severity"] == "WARN" or item["status"] == "WARN"]
    status = "MANUAL_MARKET_CONTEXT_BLOCKED" if blockers else (
        "MANUAL_MARKET_CONTEXT_WARN" if warnings else "MANUAL_MARKET_CONTEXT_PASS"
    )
    payload = {
        "generated_at_utc": clock.isoformat(),
        "status": status,
        "artifact_paths": {
            "manual_history": str(manual_history_path),
            "output": str(output_path),
            "report": str(report_path),
        },
        "sample": {
            "pair": pair,
            "manual_trades": len(trades),
            "analyzed_trades": len(analyzed),
            "coverage_pct": round(len(analyzed) / len(trades) * 100.0, 2) if trades else 0.0,
        },
        "technical_profile": profile,
        "bounded_replay_profile": bounded_profile,
        "excluded_tail_profile": excluded_tail_profile,
        "position_building_profile": position_building_profile,
        "guidance": _guidance(bounded_profile if bounded_rows else profile),
        "trade_examples": {
            "largest_winners": _trade_examples(analyzed, reverse=True),
            "largest_losers": _trade_examples(analyzed, reverse=False),
        },
        "checks": checks,
        "blockers": [item["message"] for item in blockers],
        "warnings": [item["message"] for item in warnings],
        "contract": {
            "advisory_only": True,
            "may_gate_use_of_operator_precedent_as_aggression_reason": True,
            "absence_or_mismatch_is_not_a_trade_blocker": True,
            "cannot_override": [
                "RiskEngine",
                "LiveOrderGateway",
                "gpt_trader_verifier",
                "fresh_broker_truth",
                "forecast_confidence_gate",
                "spread_and_event_gates",
                "position_close_gate_a_b",
            ],
        },
    }
    _write_json(output_path, payload)
    _write_report(report_path, payload)
    return ManualMarketContextSummary(
        output_path=output_path,
        report_path=report_path,
        status=status,
        analyzed_trades=len(analyzed),
        blockers=len(blockers),
        warnings=len(warnings),
        best_h1_alignment=(best_h1 or {}).get("bucket"),
        worst_h1_alignment=(worst_h1 or {}).get("bucket"),
    )


def _manual_trades(payload: dict[str, Any], *, pair: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in payload.get("trades") or []:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("pair") or "").upper() != pair.upper():
            continue
        open_time = _parse_dt(trade.get("open_time"))
        close_time = _parse_dt(trade.get("close_time"))
        if open_time is None or close_time is None:
            continue
        rows.append({**trade, "_open_time": open_time, "_close_time": close_time})
    return sorted(rows, key=lambda item: item["_open_time"])


def _resolve_candles(
    *,
    trades: list[dict[str, Any]],
    pair: str,
    candles_by_tf: Mapping[str, Iterable[Candle]] | None,
    client: OandaReadOnlyClient | None,
) -> tuple[dict[str, tuple[Candle, ...]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    if candles_by_tf is not None:
        resolved = {tf: tuple(candles_by_tf.get(tf) or ()) for tf in LOOKBACK_BARS_BY_TF}
        for tf, candles in resolved.items():
            checks.append(
                _check(
                    f"{tf.lower()}_candles_available",
                    "PASS" if candles else "BLOCK",
                    f"{tf} candles supplied: {len(candles)}",
                    {"timeframe": tf, "candles": len(candles), "source": "caller"},
                )
            )
        return resolved, checks

    if not trades:
        return {tf: tuple() for tf in LOOKBACK_BARS_BY_TF}, checks
    client = client or OandaReadOnlyClient()
    earliest = min(trade["_open_time"] for trade in trades)
    latest = max(trade["_close_time"] for trade in trades)
    resolved: dict[str, tuple[Candle, ...]] = {}
    for tf, lookback in LOOKBACK_BARS_BY_TF.items():
        seconds = _granularity_seconds(tf)
        start = earliest - timedelta(seconds=seconds * (lookback + 5))
        end = latest + timedelta(seconds=seconds)
        try:
            candles = _fetch_range(client, pair, tf, start, end)
            resolved[tf] = candles
            checks.append(
                _check(
                    f"{tf.lower()}_candles_available",
                    "PASS" if candles else "BLOCK",
                    f"{tf} historical candles fetched: {len(candles)}",
                    {
                        "timeframe": tf,
                        "candles": len(candles),
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit must report evidence gaps
            resolved[tf] = tuple()
            checks.append(
                _check(
                    f"{tf.lower()}_candles_available",
                    "BLOCK",
                    f"{tf} historical candles fetch failed: {exc}",
                    {"timeframe": tf, "error": str(exc)},
                )
            )
    return resolved, checks


def _fetch_range(
    client: OandaReadOnlyClient,
    pair: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> tuple[Candle, ...]:
    seconds = _granularity_seconds(timeframe)
    chunk = timedelta(seconds=seconds * OANDA_CANDLE_CHUNK_LIMIT)
    cursor = start
    by_time: dict[datetime, Candle] = {}
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        for candle in fetch_candles_between(
            pair,
            timeframe,
            time_from=cursor,
            time_to=chunk_end,
            client=client,
        ):
            if candle.complete:
                by_time[candle.timestamp_utc] = candle
        cursor = chunk_end
    return tuple(by_time[key] for key in sorted(by_time))


def _trade_context(trade: dict[str, Any], candles_by_tf: dict[str, tuple[Candle, ...]]) -> dict[str, Any] | None:
    open_time = trade["_open_time"]
    side = "LONG" if float(trade.get("units") or 0.0) > 0 else "SHORT"
    open_price = _maybe_float(trade.get("open_price"))
    hold_hours = _maybe_float(trade.get("hold_hours"))
    context: dict[str, Any] = {
        "trade_id": str(trade.get("trade_id") or ""),
        "side": side,
        "open_time": open_time.isoformat(),
        "close_time": trade["_close_time"].isoformat(),
        "session_jst": _session_jst(open_time),
        "hold_hours": hold_hours,
        "hold_bucket": _hold_bucket(hold_hours),
        "close_reason": trade.get("close_reason") or "UNKNOWN",
        "realized_pl": _maybe_float(trade.get("realized_pl")) or 0.0,
        "financing": _maybe_float(trade.get("financing")) or 0.0,
        "open_price": open_price,
    }
    usable_any = False
    for tf, required in LOOKBACK_BARS_BY_TF.items():
        prior = [candle for candle in candles_by_tf.get(tf, ()) if candle.timestamp_utc < open_time]
        series = tuple(prior[-required:])
        if len(series) < 30:
            context[f"{tf.lower()}_available"] = False
            continue
        usable_any = True
        ind = compute_indicators(str(trade.get("pair") or "USD_JPY"), tf, series)
        trend = _trend_direction(ind.linreg_slope_20)
        prefix = tf.lower()
        context[f"{prefix}_available"] = True
        context[f"{prefix}_trend"] = trend
        context[f"{prefix}_alignment"] = _alignment(side, trend, tf)
        context[f"{prefix}_rsi_14"] = _round(ind.rsi_14)
        context[f"{prefix}_adx_14"] = _round(ind.adx_14)
        context[f"{prefix}_atr_pips"] = _round(ind.atr_pips)
        context[f"{prefix}_linreg_slope_20"] = _round(ind.linreg_slope_20)
    h1_prior = [c for c in candles_by_tf.get("H1", ()) if c.timestamp_utc < open_time]
    if open_price is not None and len(h1_prior) >= 24:
        last_24 = h1_prior[-24:]
        low = min(c.low for c in last_24)
        high = max(c.high for c in last_24)
        if high > low:
            percentile = max(0.0, min(1.0, (open_price - low) / (high - low)))
            context["entry_price_percentile_24h"] = round(percentile, 4)
            context["entry_location_24h"] = _price_location(percentile)
    if context.get("h1_alignment"):
        context["side_h1_alignment"] = f"{side}_{context['h1_alignment']}"
    if context.get("entry_location_24h"):
        context["side_entry_location_24h"] = f"{side}_{context['entry_location_24h']}"
    return context if usable_any else None


def _technical_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _stats(rows),
        "by_h1_alignment": _bucket_stats(rows, "h1_alignment"),
        "by_m5_alignment": _bucket_stats(rows, "m5_alignment"),
        "by_side_h1_alignment": _bucket_stats(rows, "side_h1_alignment"),
        "by_session_jst": _bucket_stats(rows, "session_jst"),
        "by_entry_location_24h": _bucket_stats(rows, "entry_location_24h"),
        "by_side_entry_location_24h": _bucket_stats(rows, "side_entry_location_24h"),
        "by_hold_bucket": _bucket_stats(rows, "hold_bucket"),
        "by_close_reason": _bucket_stats(rows, "close_reason"),
    }


def _position_building_profile(trades: list[dict[str, Any]]) -> dict[str, Any]:
    positions = _unique_manual_positions(trades)
    clusters = _position_building_clusters(positions)
    bounded = [row for row in clusters if not row.get("unbounded_tail")]
    adverse = [row for row in bounded if int(row.get("adverse_add_count") or 0) > 0]
    return {
        "basis": (
            "same-pair same-side overlapping open/close windows reconstructed from manual "
            "OANDA exit rows; duplicate partial-exit trade_ids are counted once for entry "
            "layering and summed for realized P/L"
        ),
        "overall": _cluster_stats(clusters),
        "bounded_lt_12h_excluding_margin_closeout": _cluster_stats(bounded),
        "adverse_adds": _cluster_stats(adverse),
        "by_build_type": _cluster_bucket_stats(clusters, "build_type"),
        "bounded_by_build_type": _cluster_bucket_stats(bounded, "build_type"),
        "examples": {
            "largest_adverse_add_winners": _cluster_examples(adverse, reverse=True),
            "largest_adverse_add_losers": _cluster_examples(adverse, reverse=False),
            "largest_multi_entry_winners": _cluster_examples(
                [row for row in bounded if int(row.get("entries") or 0) > 1],
                reverse=True,
            ),
            "largest_multi_entry_losers": _cluster_examples(
                [row for row in bounded if int(row.get("entries") or 0) > 1],
                reverse=False,
            ),
        },
        "contract": {
            "advisory_only": True,
            "nanpin_is_not_live_permission": True,
            "requires_current_basket_risk_validation": True,
            "forbidden_to_use_for_unbounded_martingale": True,
        },
    }


def _unique_manual_positions(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trade: dict[str, dict[str, Any]] = {}
    for trade in trades:
        trade_id = str(trade.get("trade_id") or "")
        open_time = trade.get("_open_time")
        close_time = trade.get("_close_time")
        if not trade_id or not isinstance(open_time, datetime) or not isinstance(close_time, datetime):
            continue
        units = _maybe_float(trade.get("units"))
        open_price = _maybe_float(trade.get("open_price"))
        if units is None or units == 0 or open_price is None:
            continue
        side = "LONG" if units > 0 else "SHORT"
        existing = by_trade.get(trade_id)
        if existing is None:
            existing = {
                "trade_id": trade_id,
                "pair": str(trade.get("pair") or ""),
                "side": side,
                "units_abs": abs(units),
                "open_time": open_time,
                "close_time": close_time,
                "open_price": open_price,
                "realized_pl": 0.0,
                "financing": 0.0,
                "exit_events": 0,
                "exit_kinds": set(),
                "close_reasons": set(),
            }
            by_trade[trade_id] = existing
        existing["close_time"] = max(existing["close_time"], close_time)
        existing["realized_pl"] = float(existing["realized_pl"]) + float(trade.get("realized_pl") or 0.0)
        existing["financing"] = float(existing["financing"]) + float(trade.get("financing") or 0.0)
        existing["exit_events"] = int(existing["exit_events"]) + 1
        existing["exit_kinds"].add(str(trade.get("exit_kind") or "UNKNOWN"))
        existing["close_reasons"].add(str(trade.get("close_reason") or "UNKNOWN"))

    out: list[dict[str, Any]] = []
    for item in by_trade.values():
        item["exit_kinds"] = sorted(item["exit_kinds"])
        item["close_reasons"] = sorted(item["close_reasons"])
        out.append(item)
    return sorted(out, key=lambda item: (item["pair"], item["side"], item["open_time"], item["trade_id"]))


def _position_building_clusters(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for _, side_positions in _positions_by_pair_side(positions).items():
        current: list[dict[str, Any]] = []
        current_end: datetime | None = None
        for position in side_positions:
            open_time = position["open_time"]
            close_time = position["close_time"]
            if not current or current_end is None or open_time > current_end:
                if current:
                    clusters.append(_summarize_position_cluster(current))
                current = [position]
                current_end = close_time
                continue
            current.append(position)
            current_end = max(current_end, close_time)
        if current:
            clusters.append(_summarize_position_cluster(current))
    return sorted(clusters, key=lambda item: item["start_time"])


def _positions_by_pair_side(positions: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for position in positions:
        key = (str(position.get("pair") or ""), str(position.get("side") or ""))
        grouped.setdefault(key, []).append(position)
    return {
        key: sorted(items, key=lambda item: (item["open_time"], item["trade_id"]))
        for key, items in grouped.items()
    }


def _summarize_position_cluster(positions: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(positions, key=lambda item: (item["open_time"], item["trade_id"]))
    first = ordered[0]
    side = str(first.get("side") or "")
    pip_factor = 100 if str(first.get("pair") or "").endswith("_JPY") else 10000
    final_weighted_avg = float(first.get("open_price") or 0.0)
    add_rows: list[dict[str, Any]] = []
    adverse_count = 0
    pyramid_count = 0
    flat_count = 0
    for idx, position in enumerate(ordered[1:], start=1):
        active_before = _active_positions_at(ordered[:idx], position["open_time"])
        weighted_units, weighted_avg = _weighted_open_price(active_before)
        if weighted_units <= 0:
            weighted_units = float(ordered[idx - 1].get("units_abs") or 0.0)
            weighted_avg = float(ordered[idx - 1].get("open_price") or 0.0)
        units = float(position.get("units_abs") or 0.0)
        price = float(position.get("open_price") or 0.0)
        adverse_move_pips = _adverse_add_pips(side, weighted_avg, price, pip_factor)
        if adverse_move_pips > 0:
            relation = "AVERAGE_INTO_ADVERSE"
            adverse_count += 1
        elif adverse_move_pips < 0:
            relation = "PYRAMID_WITH_MOVE"
            pyramid_count += 1
        else:
            relation = "FLAT_ADD"
            flat_count += 1
        next_units = weighted_units + units
        next_avg = ((weighted_avg * weighted_units) + (price * units)) / next_units if next_units else weighted_avg
        final_weighted_avg = next_avg
        add_rows.append(
            {
                "trade_id": position.get("trade_id"),
                "open_time": position["open_time"].isoformat(),
                "open_price": price,
                "units": units,
                "relation": relation,
                "adverse_move_pips": round(adverse_move_pips, 2),
                "avg_before": round(weighted_avg, 5),
                "avg_after": round(next_avg, 5),
            }
        )

    build_type = _cluster_build_type(adverse_count, pyramid_count, flat_count, len(ordered))
    start_time = min(item["open_time"] for item in ordered)
    end_time = max(item["close_time"] for item in ordered)
    hold_hours = (end_time - start_time).total_seconds() / 3600.0
    close_reasons = sorted({reason for item in ordered for reason in item.get("close_reasons", [])})
    return {
        "cluster_id": f"{first.get('pair')}:{side}:{start_time.isoformat()}",
        "pair": first.get("pair"),
        "side": side,
        "build_type": build_type,
        "entries": len(ordered),
        "trade_ids": [item.get("trade_id") for item in ordered],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "session_jst": _session_jst(start_time),
        "hold_hours": round(hold_hours, 3),
        "hold_bucket": _hold_bucket(hold_hours),
        "realized_pl": round(sum(float(item.get("realized_pl") or 0.0) for item in ordered), 1),
        "financing": round(sum(float(item.get("financing") or 0.0) for item in ordered), 1),
        "initial_units": float(first.get("units_abs") or 0.0),
        "max_units_abs": round(_max_active_units_abs(ordered), 2),
        "initial_price": first.get("open_price"),
        "final_weighted_avg": round(final_weighted_avg, 5),
        "adverse_add_count": adverse_count,
        "pyramid_add_count": pyramid_count,
        "flat_add_count": flat_count,
        "adds": add_rows,
        "close_reasons": close_reasons,
        "unbounded_tail": hold_hours >= HOLD_BUCKET_HOURS[2] or "MARKET_ORDER_MARGIN_CLOSEOUT" in close_reasons,
    }


def _active_positions_at(positions: list[dict[str, Any]], at: datetime) -> list[dict[str, Any]]:
    return [item for item in positions if item["open_time"] <= at <= item["close_time"]]


def _weighted_open_price(positions: list[dict[str, Any]]) -> tuple[float, float]:
    units = sum(float(item.get("units_abs") or 0.0) for item in positions)
    if units <= 0:
        return 0.0, 0.0
    weighted = sum(float(item.get("open_price") or 0.0) * float(item.get("units_abs") or 0.0) for item in positions)
    return units, weighted / units


def _max_active_units_abs(positions: list[dict[str, Any]]) -> float:
    max_units = 0.0
    for at in sorted({item["open_time"] for item in positions}):
        units, _ = _weighted_open_price(_active_positions_at(positions, at))
        max_units = max(max_units, units)
    return max_units


def _adverse_add_pips(side: str, current_avg: float, add_price: float, pip_factor: int) -> float:
    if side == "LONG":
        return (current_avg - add_price) * pip_factor
    if side == "SHORT":
        return (add_price - current_avg) * pip_factor
    return 0.0


def _cluster_build_type(adverse_count: int, pyramid_count: int, flat_count: int, entries: int) -> str:
    if entries <= 1:
        return "SINGLE_ENTRY"
    if adverse_count and not pyramid_count:
        return "AVERAGE_INTO_ADVERSE"
    if pyramid_count and not adverse_count:
        return "PYRAMID_WITH_MOVE"
    if adverse_count and pyramid_count:
        return "MIXED_POSITION_BUILD"
    if flat_count:
        return "FLAT_MULTI_ENTRY"
    return "MULTI_ENTRY_UNKNOWN"


def _cluster_bucket_stats(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "UNKNOWN"), []).append(row)
    out = []
    for bucket, items in grouped.items():
        item = _cluster_stats(items)
        item["bucket"] = bucket
        out.append(item)
    return sorted(out, key=lambda item: float(item["net_jpy"]), reverse=True)


def _cluster_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pls = [float(row.get("realized_pl") or 0.0) for row in rows]
    wins = [value for value in pls if value > 0]
    losses = [value for value in pls if value < 0]
    entries = [int(row.get("entries") or 0) for row in rows]
    holds = [float(row.get("hold_hours") or 0.0) for row in rows]
    adverse_moves = [
        float(add.get("adverse_move_pips") or 0.0)
        for row in rows
        for add in row.get("adds", [])
        if str(add.get("relation") or "") == "AVERAGE_INTO_ADVERSE"
    ]
    return {
        "clusters": len(rows),
        "multi_entry_clusters": sum(1 for value in entries if value > 1),
        "entries": sum(entries),
        "net_jpy": round(sum(pls), 1),
        "win_rate": round(len(wins) / len(pls), 3) if pls else None,
        "avg_win": round(statistics.mean(wins), 1) if wins else None,
        "avg_loss": round(statistics.mean(losses), 1) if losses else None,
        "payoff": round(abs(statistics.mean(wins) / statistics.mean(losses)), 2) if wins and losses else None,
        "expectancy_jpy": round(statistics.mean(pls), 1) if pls else None,
        "median_entries": round(statistics.median(entries), 2) if entries else None,
        "max_entries": max(entries) if entries else 0,
        "median_hold_hours": round(statistics.median(holds), 2) if holds else None,
        "adverse_adds": sum(int(row.get("adverse_add_count") or 0) for row in rows),
        "pyramid_adds": sum(int(row.get("pyramid_add_count") or 0) for row in rows),
        "avg_adverse_add_pips": round(statistics.mean(adverse_moves), 2) if adverse_moves else None,
    }


def _cluster_examples(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda item: float(item.get("realized_pl") or 0.0), reverse=reverse)[:5]
    keys = (
        "cluster_id",
        "side",
        "build_type",
        "entries",
        "trade_ids",
        "start_time",
        "session_jst",
        "hold_hours",
        "realized_pl",
        "initial_price",
        "final_weighted_avg",
        "adverse_add_count",
        "pyramid_add_count",
        "close_reasons",
    )
    return [{key: item.get(key) for key in keys if key in item} for item in selected]


def _is_unbounded_tail(row: dict[str, Any]) -> bool:
    hold = _maybe_float(row.get("hold_hours"))
    if hold is not None and hold >= HOLD_BUCKET_HOURS[2]:
        return True
    return str(row.get("close_reason") or "").upper() == "MARKET_ORDER_MARGIN_CLOSEOUT"


def _bucket_stats(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = str(row.get(key) or "UNKNOWN")
        grouped.setdefault(bucket, []).append(row)
    out = []
    for bucket, items in grouped.items():
        item = _stats(items)
        item["bucket"] = bucket
        out.append(item)
    return sorted(out, key=lambda item: float(item["net_jpy"]), reverse=True)


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pls = [float(row.get("realized_pl") or 0.0) for row in rows]
    wins = [value for value in pls if value > 0]
    losses = [value for value in pls if value < 0]
    holds = [float(row["hold_hours"]) for row in rows if _maybe_float(row.get("hold_hours")) is not None]
    h1_adx = [float(row["h1_adx_14"]) for row in rows if _maybe_float(row.get("h1_adx_14")) is not None]
    pcts = [
        float(row["entry_price_percentile_24h"])
        for row in rows
        if _maybe_float(row.get("entry_price_percentile_24h")) is not None
    ]
    return {
        "trades": len(rows),
        "net_jpy": round(sum(pls), 1),
        "win_rate": round(len(wins) / len(pls), 3) if pls else None,
        "avg_win": round(statistics.mean(wins), 1) if wins else None,
        "avg_loss": round(statistics.mean(losses), 1) if losses else None,
        "payoff": round(abs(statistics.mean(wins) / statistics.mean(losses)), 2) if wins and losses else None,
        "expectancy_jpy": round(statistics.mean(pls), 1) if pls else None,
        "median_hold_hours": round(statistics.median(holds), 2) if holds else None,
        "avg_h1_adx": round(statistics.mean(h1_adx), 1) if h1_adx else None,
        "median_entry_price_percentile_24h": round(statistics.median(pcts), 3) if pcts else None,
    }


def _guidance(profile: dict[str, Any]) -> dict[str, Any]:
    best_h1 = _best_bucket(profile["by_h1_alignment"])
    worst_h1 = _worst_bucket(profile["by_h1_alignment"])
    best_session = _preferred_session_bucket(profile)
    worst_hold = _worst_bucket(profile["by_hold_bucket"])
    return {
        "basis": "bounded_replay_lt_12h_excluding_margin_closeout",
        "prefer_when_citing_precedent": {
            "h1_alignment": (best_h1 or {}).get("bucket"),
            "session_jst": (best_session or {}).get("bucket"),
        },
        "require_extra_current_reason_when_conflicting": {
            "h1_alignment": (worst_h1 or {}).get("bucket"),
            "hold_bucket": (worst_hold or {}).get("bucket"),
        },
        "operator_precedent_usage_gate": (
            "A current lane may cite the 2025 manual precedent as an aggression/ranking reason only when "
            "its pair/direction/session and H1/M5 technical context are comparable; otherwise cite the "
            "current deterministic edge instead."
        ),
    }


def _preferred_session_bucket(profile: dict[str, Any]) -> dict[str, Any] | None:
    rows = list(profile.get("by_session_jst") or [])
    if not rows:
        return None
    total = int((profile.get("overall") or {}).get("trades") or 0)
    overall_win_rate = _maybe_float((profile.get("overall") or {}).get("win_rate"))
    # Sample-size guard for descriptive precedent selection. The floor prevents
    # a handful of large wins from defining "the" session; 10% keeps the guard
    # proportional when auditing smaller synthetic/test samples. Not a market
    # gate and never used by RiskEngine.
    min_trades = min(total, max(30, int(total * 0.10))) if total else 0
    eligible = [
        row
        for row in rows
        if int(row.get("trades") or 0) >= min_trades
        and float(row.get("net_jpy") or 0.0) > 0
        and (
            overall_win_rate is None
            or (_maybe_float(row.get("win_rate")) or 0.0) >= overall_win_rate
        )
    ]
    if eligible:
        return sorted(eligible, key=lambda item: float(item.get("net_jpy") or 0.0), reverse=True)[0]
    positives = [row for row in rows if float(row.get("net_jpy") or 0.0) > 0]
    return sorted(positives or rows, key=lambda item: float(item.get("net_jpy") or 0.0), reverse=True)[0]


def _trade_examples(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda item: float(item.get("realized_pl") or 0.0), reverse=reverse)[:5]
    keys = (
        "trade_id",
        "side",
        "open_time",
        "session_jst",
        "hold_hours",
        "realized_pl",
        "close_reason",
        "h1_trend",
        "h1_alignment",
        "m5_trend",
        "m5_alignment",
        "entry_price_percentile_24h",
        "entry_location_24h",
    )
    return [{key: item.get(key) for key in keys if key in item} for item in selected]


def _best_bucket(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _worst_bucket(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return sorted(rows, key=lambda item: float(item["net_jpy"]))[0] if rows else None


def _trend_direction(slope: float | None) -> str:
    if slope is None:
        return "UNKNOWN"
    if slope > 0:
        return "UP"
    if slope < 0:
        return "DOWN"
    return "FLAT"


def _alignment(side: str, trend: str, timeframe: str) -> str:
    if trend == "UNKNOWN" or trend == "FLAT":
        return f"{timeframe}_FLAT_OR_UNKNOWN"
    if (side == "LONG" and trend == "UP") or (side == "SHORT" and trend == "DOWN"):
        return f"WITH_{timeframe}_TREND"
    return f"AGAINST_{timeframe}_TREND"


def _price_location(percentile: float) -> str:
    # Terciles describe where entries occurred inside the prior 24h range.
    # They are not execution thresholds and do not affect live permission.
    if percentile < 1.0 / 3.0:
        return "LOWER_THIRD_24H"
    if percentile > 2.0 / 3.0:
        return "UPPER_THIRD_24H"
    return "MIDDLE_THIRD_24H"


def _hold_bucket(hours: float | None) -> str:
    if hours is None:
        return "UNKNOWN"
    if hours < HOLD_BUCKET_HOURS[0]:
        return "<30M"
    if hours < HOLD_BUCKET_HOURS[1]:
        return "30M_2H"
    if hours < HOLD_BUCKET_HOURS[2]:
        return "2H_12H"
    return "GE_12H"


def _session_jst(value: datetime) -> str:
    hour = value.astimezone(timezone.utc).hour
    if 0 <= hour < 6:
        return "TOKYO"
    if 6 <= hour < 12:
        return "LONDON_AM"
    if 12 <= hour < 18:
        return "NY_OVERLAP"
    return "OFF_HOURS"


def _granularity_seconds(timeframe: str) -> int:
    return {"M5": 300, "H1": 3600}[timeframe]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    raw_profile = payload["technical_profile"]
    bounded_profile = payload["bounded_replay_profile"]
    excluded_tail = payload["excluded_tail_profile"]
    position_building = payload["position_building_profile"]
    guidance = payload["guidance"]
    building_overall = position_building.get("overall") or {}
    adverse_adds = position_building.get("adverse_adds") or {}
    lines = [
        "# Manual Market Context Audit",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Pair: `{payload['sample']['pair']}`",
        f"- Analyzed trades: `{payload['sample']['analyzed_trades']}` / `{payload['sample']['manual_trades']}` (`{payload['sample']['coverage_pct']}`%)",
        f"- Guidance basis: `{guidance.get('basis')}`",
        f"- Best H1 alignment bucket: `{guidance['prefer_when_citing_precedent'].get('h1_alignment')}`",
        f"- Best session bucket: `{guidance['prefer_when_citing_precedent'].get('session_jst')}`",
        f"- Conflict bucket requiring extra current reason: `{guidance['require_extra_current_reason_when_conflicting'].get('h1_alignment')}`",
        f"- Position-building clusters: `{building_overall.get('multi_entry_clusters')}` multi-entry / `{building_overall.get('clusters')}` total",
        f"- Averaging-into-adverse clusters: `{adverse_adds.get('clusters')}` net `{adverse_adds.get('net_jpy')}` JPY",
        "",
        "## Bounded H1 Alignment",
        "",
        "Bounded replay excludes >=12h holds and margin-closeout exits, because those are the same unbounded carry tail this runtime must avoid.",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bounded_profile["by_h1_alignment"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Bounded Side x H1 Alignment",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bounded_profile["by_side_h1_alignment"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Bounded Side x 24h Location",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bounded_profile["by_side_entry_location_24h"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Raw H1 Alignment",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw_profile["by_h1_alignment"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Bounded Session",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bounded_profile["by_session_jst"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Position Building",
        "",
        "Position-building clusters reconstruct overlapping same-pair, same-side manual exposure from OANDA exit rows. Duplicate partial-exit trade ids count once as an entry layer, while realized P/L is summed.",
        "",
        "| bucket | clusters | multi | entries | net JPY | win rate | expectancy | median entries | max entries | adverse adds | pyramid adds | avg adverse add pips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in position_building["bounded_by_build_type"]:
        lines.append(_cluster_bucket_line(row))
    lines += [
        "",
        "## Averaging Into Adverse Examples",
        "",
        "| side | entries | P/L JPY | hold h | initial | final avg | adverse adds | trade ids |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in (position_building.get("examples") or {}).get("largest_adverse_add_winners", [])[:5]:
        lines.append(_cluster_example_line(row))
    lines += [
        "",
        "## Excluded Tail",
        "",
        "| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in excluded_tail["by_hold_bucket"]:
        lines.append(_bucket_line(row))
    lines += [
        "",
        "## Contract",
        "",
        "- Advisory only: this audit gates only whether the 2025 manual precedent may be cited as an aggression/ranking reason.",
        "- It cannot override RiskEngine, LiveOrderGateway, forecast, spread, event, broker-truth, or close Gate A/B checks.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bucket_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row.get('bucket')}` | `{row.get('trades')}` | `{row.get('net_jpy')}` | "
        f"`{row.get('win_rate')}` | `{row.get('expectancy_jpy')}` | "
        f"`{row.get('median_hold_hours')}` | `{row.get('avg_h1_adx')}` |"
    )


def _cluster_bucket_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row.get('bucket')}` | `{row.get('clusters')}` | `{row.get('multi_entry_clusters')}` | "
        f"`{row.get('entries')}` | `{row.get('net_jpy')}` | `{row.get('win_rate')}` | "
        f"`{row.get('expectancy_jpy')}` | `{row.get('median_entries')}` | `{row.get('max_entries')}` | "
        f"`{row.get('adverse_adds')}` | `{row.get('pyramid_adds')}` | `{row.get('avg_adverse_add_pips')}` |"
    )


def _cluster_example_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row.get('side')}` | `{row.get('entries')}` | `{row.get('realized_pl')}` | "
        f"`{row.get('hold_hours')}` | `{row.get('initial_price')}` | `{row.get('final_weighted_avg')}` | "
        f"`{row.get('adverse_add_count')}` | `{', '.join(str(item) for item in row.get('trade_ids') or [])}` |"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(
    check_name: str,
    status: str,
    message: str,
    evidence: dict[str, Any],
    *,
    severity: str | None = None,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity or ("BLOCK" if status == "BLOCK" else "INFO"),
        "message": message,
        "evidence": evidence,
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "top-level JSON is not an object"
    return payload, None


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        if "." in text:
            head, rest = text.split(".", 1)
            digit_count = 0
            while digit_count < len(rest) and rest[digit_count].isdigit():
                digit_count += 1
            fraction = rest[:digit_count][:6]
            text = f"{head}.{fraction}{rest[digit_count:]}"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _maybe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None
