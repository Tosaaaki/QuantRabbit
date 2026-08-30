"""Causal audit helpers for one fast-bot shock episode.

The module joins already-recorded shadow proposals to their exact S5 bid/ask
outcomes.  It never substitutes candle direction for a missing strategy label
and it has no broker client or mutation surface.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping


def _utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _profit_factor(values: Iterable[float]) -> float | None:
    rows = list(values)
    wins = sum(value for value in rows if value > 0.0)
    losses = -sum(value for value in rows if value < 0.0)
    return round(wins / losses, 6) if losses > 0.0 else None


def _maximum_loss_streak(values: Iterable[float]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value < 0.0 else 0
        best = max(best, current)
    return best


def trade_metrics(rows: Iterable[Mapping[str, Any]], *, scale: float = 1.0) -> dict[str, Any]:
    source = list(rows)
    selected = [row for row in source if row.get("filled") is True]
    values = [float(row.get("realized_pips") or 0.0) * scale for row in selected]
    ordered = sorted(values)
    p05 = None
    if ordered:
        position = max(0, int((len(ordered) - 1) * 0.05))
        p05 = round(ordered[position], 6)
    return {
        "proposals": len(source),
        "filled_trades": len(values),
        "net_pips": round(sum(values), 6),
        "profit_factor": _profit_factor(values),
        "hit_rate": round(sum(value > 0.0 for value in values) / len(values), 6)
        if values
        else None,
        "p05_trade_pips": p05,
        "maximum_loss_streak": _maximum_loss_streak(values),
    }


def _joined_rows(
    signals: Iterable[Mapping[str, Any]], outcomes: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    outcome_by_signal = {
        str(row.get("signal_id")): row
        for row in outcomes
        if row.get("signal_id") is not None
    }
    joined: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        outcome = outcome_by_signal.get(signal_id)
        joined.append(
            {
                "signal_id": signal_id,
                "generated_at_utc": signal.get("generated_at_utc"),
                "pair": signal.get("pair"),
                "side": signal.get("side"),
                "method": signal.get("method"),
                "strategy_id": signal.get("strategy_id"),
                "regime_score": signal.get("regime_score"),
                "entry": signal.get("entry"),
                "take_profit_pips": signal.get("take_profit_pips"),
                "stop_loss_pips": signal.get("stop_loss_pips"),
                "spread_pips": signal.get("spread_pips"),
                "m5_atr_pips": signal.get("m5_atr_pips"),
                "quote_bid": signal.get("quote_bid"),
                "quote_ask": signal.get("quote_ask"),
                "quote_timestamp_utc": signal.get("quote_timestamp_utc"),
                "shadow_only": signal.get("shadow_only"),
                "live_permission": signal.get("live_permission"),
                "outcome_retained": outcome is not None,
                "filled": outcome.get("filled") if outcome else None,
                "fill_at_utc": outcome.get("fill_at_utc") if outcome else None,
                "exit_at_utc": outcome.get("exit_at_utc") if outcome else None,
                "exit_reason": outcome.get("exit_reason") if outcome else None,
                "realized_pips": outcome.get("realized_pips") if outcome else None,
                "ambiguous_same_s5": outcome.get("ambiguous_same_s5") if outcome else None,
                "truth_request_coverage_proved": outcome.get("truth_request_coverage_proved")
                if outcome
                else None,
                "truth_grid_slot_count": outcome.get("truth_grid_slot_count") if outcome else None,
                "truth_no_tick_slot_count": outcome.get("truth_no_tick_slot_count")
                if outcome
                else None,
            }
        )
    return joined


def _slice(
    rows: Iterable[Mapping[str, Any]], start: datetime, end: datetime
) -> list[Mapping[str, Any]]:
    result = []
    for row in rows:
        generated = row.get("generated_at_utc")
        if generated is None:
            continue
        at = _utc(generated)
        if start <= at < end:
            result.append(row)
    return result


def _summary(rows: list[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        result.append(
            {
                **dict(zip(keys, group_key)),
                **trade_metrics(group_rows),
                "unfilled": sum(row.get("filled") is False for row in group_rows),
                "missing_outcome": sum(row.get("outcome_retained") is False for row in group_rows),
            }
        )
    return result


def audit_shock_episode(
    *,
    signals: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
    window_start_utc: datetime,
    window_end_utc: datetime,
    shock_at_utc: datetime,
    shock_pair: str,
    shock_direction: str,
    historical_episode_class: str,
) -> dict[str, Any]:
    """Join a bounded incident window and compare causal, no-order arms."""

    start = window_start_utc.astimezone(timezone.utc)
    end = window_end_utc.astimezone(timezone.utc)
    shock_at = shock_at_utc.astimezone(timezone.utc)
    signal_rows = _slice(signals, start, end)
    joined = _joined_rows(signal_rows, outcomes)
    shock_pair = shock_pair.upper()
    shock_direction = shock_direction.upper()
    for row in joined:
        signal_direction = "UP" if row.get("side") == "LONG" else "DOWN"
        alignment = "ALIGNED" if signal_direction == shock_direction else "COUNTERTREND"
        row["signal_direction"] = signal_direction
        row["side_relative_alignment"] = alignment
        row["regime_transition_mismatch"] = bool(
            row.get("pair") == shock_pair
            and row.get("method") == "RANGE_ROTATION"
            and alignment == "COUNTERTREND"
        )
        row["strategy_label_source"] = (
            "ACTUAL_SIGNAL_LEDGER"
            if row.get("strategy_id") is not None and row.get("method") is not None
            else "MISSING_NOT_PROXIED"
        )

    exact_window = [
        row
        for row in joined
        if abs((_utc(row["generated_at_utc"]) - shock_at).total_seconds()) <= 30
    ]
    usd_exact = [row for row in exact_window if row.get("pair") == "USD_JPY"]
    if not usd_exact:
        usd_reason = "NO_SIGNAL_IN_EXACT_EVENT_WINDOW"
    elif any(row.get("filled") is True for row in usd_exact):
        usd_reason = "VIRTUAL_FILL_OCCURRED_NOT_ABSENT"
    elif all(
        row.get("outcome_retained") is True
        and row.get("filled") is False
        and row.get("exit_reason") == "UNFILLED"
        and row.get("truth_request_coverage_proved") is True
        for row in usd_exact
    ):
        usd_reason = "PASSIVE_LIMIT_NOT_TOUCHED_WITHIN_TTL"
    elif any(row.get("outcome_retained") is False for row in usd_exact):
        usd_reason = "OUTCOME_NOT_RETAINED_UNCONFIRMED"
    else:
        usd_reason = "UNRESOLVED_FROM_RETAINED_FIELDS"

    pair_rows = [row for row in joined if row.get("pair") == shock_pair]
    freeze_until = shock_at + timedelta(minutes=5)
    frozen_ids = {
        row["signal_id"]
        for row in pair_rows
        if shock_at <= _utc(row["generated_at_utc"]) < freeze_until
    }
    mismatch_ids = {
        row["signal_id"]
        for row in pair_rows
        if _utc(row["generated_at_utc"]) >= shock_at
        and row.get("regime_transition_mismatch") is True
    }
    baseline = pair_rows
    freeze_arm = [row for row in pair_rows if row["signal_id"] not in frozen_ids]
    mismatch_arm = [row for row in pair_rows if row["signal_id"] not in mismatch_ids]
    trend_arm = [
        row
        for row in pair_rows
        if _utc(row["generated_at_utc"]) >= freeze_until
        and row.get("method") == "TREND_CONTINUATION"
        and row.get("side_relative_alignment") == "ALIGNED"
    ]
    reversal_arm = [
        row
        for row in pair_rows
        if historical_episode_class == "V_REVERSAL"
        and _utc(row["generated_at_utc"]) >= freeze_until
        and row.get("side_relative_alignment") == "COUNTERTREND"
    ]
    whipsaw_arm = [] if historical_episode_class == "WHIPSAW" else freeze_arm

    baseline_metrics = trade_metrics(baseline)
    arms = {
        "baseline": baseline_metrics,
        "shock_freeze_5m": trade_metrics(freeze_arm),
        "side_relative_regime_transition_veto": trade_metrics(mismatch_arm),
        "trend_aligned_continuation_after_5m_half_size": trade_metrics(
            trend_arm, scale=0.5
        ),
        "v_reversal_confirmed_only": trade_metrics(reversal_arm),
        "whipsaw_freeze": trade_metrics(whipsaw_arm),
        "bot_owned_50pct_staged_drain_proxy": trade_metrics(baseline, scale=0.5),
        "catastrophic_stop_plus_structure_exit": {
            "status": "NOT_IDENTIFIABLE_FROM_PROPOSAL_OUTCOME_LEDGER_ONLY",
            "reason": "The retained outcome row has the original fixed geometry but not the complete causal S5 path needed to rescore a new exit architecture.",
        },
    }
    for name, metrics in arms.items():
        if not isinstance(metrics, dict) or metrics.get("net_pips") is None:
            continue
        metrics["loss_avoidance_vs_baseline_pips"] = round(
            float(metrics["net_pips"]) - float(baseline_metrics["net_pips"]), 6
        )
        pf = metrics.get("profit_factor")
        metrics["profit_creating"] = bool(pf is not None and float(pf) > 1.0)
        metrics["live_promotion_allowed"] = False

    exact_eurusd = [row for row in exact_window if row.get("pair") == shock_pair]
    mismatch_rows = [row for row in pair_rows if row.get("regime_transition_mismatch")]
    return {
        "contract": "QR_FAST_BOT_SHOCK_CAUSAL_AUDIT_V1",
        "window": {"from_utc": start.isoformat(), "to_utc": end.isoformat()},
        "shock": {
            "pair": shock_pair,
            "at_utc": shock_at.isoformat(),
            "direction": shock_direction,
            "historical_episode_class": historical_episode_class,
        },
        "timeline": joined,
        "exact_event_eurusd": exact_eurusd,
        "exact_event_usdjpy": usd_exact,
        "usdjpy_participation_reason": usd_reason,
        "usdjpy_real_order_reason": "EXECUTION_AUTHORITY_NONE_SHADOW_ONLY",
        "strategy_label_coverage": {
            "proposals": len(joined),
            "labeled": sum(row["strategy_label_source"] == "ACTUAL_SIGNAL_LEDGER" for row in joined),
            "missing": sum(row["strategy_label_source"] != "ACTUAL_SIGNAL_LEDGER" for row in joined),
            "price_proxy_used": False,
        },
        "regime_transition_mismatch": {
            "count": len(mismatch_rows),
            "metrics": trade_metrics(mismatch_rows),
            "definition": "RANGE_ROTATION proposal opposite the detected shock direction",
            "side_specific_rule_used": False,
        },
        "by_pair_method_side": _summary(joined, ("pair", "method", "side")),
        "arms_same_proposal_stream": arms,
        "llm_receipt": {
            "exact_historical_receipt_retained_in_signal_ledger": False,
            "order_fields_inferred_from_llm": False,
            "permitted_scope": ["REGIME", "ALLOWED_STRATEGY_IDS", "RISK_BUDGET_CAP", "EXPIRY"],
        },
        "execution_authority": "NONE",
        "gateway_invocations": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
