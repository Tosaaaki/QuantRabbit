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
    guidance = payload["guidance"]
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
